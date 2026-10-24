#!/usr/bin/env python3
"""Nodo ROS 2 para detectar cruces tipo zebra con vision clasica.

Filtra rayas candidatas y valida geometria del conjunto para reducir falsos
positivos antes de publicar /zebra_detected y /zebra_stripes.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Int32
from sensor_msgs.msg import Image, CompressedImage

def fresh_qos(depth=1):
    """QoS de mínima latencia: best-effort, solo el frame más nuevo."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )

class ZebraDetector(Node):
    def __init__(self):
        super().__init__('zebra_detector')
        self.get_logger().info('Zebra detector node started (geometría + debounce).')

        self.bridge     = CvBridge()
        self.last_frame = None

        self.zebra_state   = False
        self._on_counter   = 0
        self._off_counter  = 0

        self.declare_parameter('compressed_topic', '/video_source/compressed')
        self.declare_parameter('zebra_topic',      '/zebra_detected')
        self.declare_parameter('stripes_topic',    '/zebra_stripes')

        self.declare_parameter('roi_top_ratio',    0.70)
        self.declare_parameter('roi_bottom_ratio', 0.85)

        self.declare_parameter('roi_left_ratio',   0.15)   # 0.0 = desde el borde izquierdo
        self.declare_parameter('roi_right_ratio',  0.85)   # 1.0 = hasta el borde derecho

        self.declare_parameter('adaptive_block_size', 51)
        self.declare_parameter('adaptive_c',          10)

        self.declare_parameter('min_stripe_area',   150)
        self.declare_parameter('max_stripe_area',  6000)
        self.declare_parameter('min_aspect_ratio',  1.8)

        # Validación geométrica del conjunto
        self.declare_parameter('min_stripes',             3)
        self.declare_parameter('perp_spread_ratio_max',   0.35)
        self.declare_parameter('spacing_cv_max',          0.55)

        self.declare_parameter('debounce_on',   3)
        self.declare_parameter('debounce_off',  5)

        self.declare_parameter('debug_view', False)

        compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        zebra_topic      = self.get_parameter('zebra_topic').get_parameter_value().string_value
        stripes_topic    = self.get_parameter('stripes_topic').get_parameter_value().string_value

        self._cache_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.sub = self.create_subscription(
            CompressedImage,
            compressed_topic,
            self.camera_callback,
            fresh_qos(1),
        )
        self.get_logger().info(f'Suscrito a {compressed_topic} (JPEG, fresh).')

        self.zebra_pub    = self.create_publisher(Bool,  zebra_topic,   10)
        self.stripes_pub  = self.create_publisher(Int32, stripes_topic, 10)
        self.dbg_vis_pub  = self.create_publisher(Image, '/debug/zebra_vis',  1)
        self.dbg_mask_pub = self.create_publisher(Image, '/debug/zebra_mask', 1)

        self.timer = self.create_timer(0.05, self.timer_callback)

    def _cache_params(self):
        self.p_roi_top_ratio    = float(self.get_parameter('roi_top_ratio').value)
        self.p_roi_bottom_ratio = float(self.get_parameter('roi_bottom_ratio').value)

        self.p_roi_left_ratio   = float(self.get_parameter('roi_left_ratio').value)
        self.p_roi_right_ratio  = float(self.get_parameter('roi_right_ratio').value)

        block = int(self.get_parameter('adaptive_block_size').value)
        self.p_adaptive_block_size = block if block % 2 == 1 else block + 1
        self.p_adaptive_c          = int(self.get_parameter('adaptive_c').value)

        self.p_min_stripe_area  = int(self.get_parameter('min_stripe_area').value)
        self.p_max_stripe_area  = int(self.get_parameter('max_stripe_area').value)
        self.p_min_aspect_ratio = float(self.get_parameter('min_aspect_ratio').value)

        self.p_min_stripes           = int(self.get_parameter('min_stripes').value)
        self.p_perp_spread_ratio_max = float(self.get_parameter('perp_spread_ratio_max').value)
        self.p_spacing_cv_max        = float(self.get_parameter('spacing_cv_max').value)

        self.p_debounce_on  = int(self.get_parameter('debounce_on').value)
        self.p_debounce_off = int(self.get_parameter('debounce_off').value)

        self.p_debug_view = bool(self.get_parameter('debug_view').value)

    def _on_params_changed(self, params):
        self._cache_params()
        return SetParametersResult(successful=True)

    def camera_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('imdecode devolvió None.')
            return
        self.last_frame = frame

    def _binarize(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.p_adaptive_block_size,
            C=self.p_adaptive_c,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def _find_stripe_candidates(self, binary):
        """
        Devuelve lista de candidatos:
          {'centroid': (cx, cy), 'area': float, 'bbox': (x,y,w,h), 'contour': cnt}
        Filtra por área y aspect ratio.
        """
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.p_min_stripe_area <= area <= self.p_max_stripe_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue

            aspect = max(w, h) / float(min(w, h))
            if aspect < self.p_min_aspect_ratio:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            candidates.append({
                'centroid': (cx, cy),
                'area':     area,
                'bbox':     (x, y, w, h),
                'contour':  cnt,
            })

        return candidates

    # Validación geométrica del conjunto
    def _validate_zebra(self, candidates):
        """
        Decide si el conjunto de candidatos forma una zebra real.

        Criterios:
          1. nº de rayas >= min_stripes
          2. colinealidad: dispersión perpendicular pequeña vs eje de arreglo
          3. espaciado regular: bajo coeficiente de variación de los gaps

        Devuelve:
          (is_zebra: bool, info: dict)
        """
        info = {
            'n': len(candidates),
            'cv': None,
            'perp_ratio': None,
            'reason': '',
        }

        if len(candidates) < self.p_min_stripes:
            info['reason'] = 'pocas rayas'
            return False, info

        pts = np.array([c['centroid'] for c in candidates], dtype=float)

        std_x = pts[:, 0].std()
        std_y = pts[:, 1].std()

        # El eje de arreglo es el de mayor dispersión.
        # El eje perpendicular debe tener poca dispersión.
        if std_y >= std_x:
            arr_axis = pts[:, 1]
            arr_std  = std_y
            perp_std = std_x
        else:
            arr_axis = pts[:, 0]
            arr_std  = std_x
            perp_std = std_y

        if arr_std < 1e-6:
            info['reason'] = 'sin dispersión en eje de arreglo'
            return False, info

        perp_ratio = perp_std / arr_std
        info['perp_ratio'] = round(perp_ratio, 3)

        if perp_ratio > self.p_perp_spread_ratio_max:
            info['reason'] = 'no colineales'
            return False, info

        # Espaciado regular
        sorted_axis = np.sort(arr_axis)
        gaps = np.diff(sorted_axis)

        if len(gaps) < 1:
            info['reason'] = 'gaps insuficientes'
            return False, info

        mean_gap = gaps.mean()
        if mean_gap < 1e-6:
            info['reason'] = 'gap medio nulo'
            return False, info

        cv = gaps.std() / mean_gap
        info['cv'] = round(cv, 3)

        if cv > self.p_spacing_cv_max:
            info['reason'] = 'espaciado irregular'
            return False, info

        info['reason'] = 'OK'
        return True, info

    def _update_debounce(self, raw_detected):
        if raw_detected:
            self._on_counter += 1
            self._off_counter = 0

            if self._on_counter >= self.p_debounce_on:
                self.zebra_state = True
        else:
            self._off_counter += 1
            self._on_counter = 0

            if self._off_counter >= self.p_debounce_off:
                self.zebra_state = False

        return self.zebra_state

    def _pub_img(self, publisher, img, encoding='bgr8'):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error publicando imagen de debug: {e}')

    def timer_callback(self):
        if self.last_frame is None:
            return

        frame = self.last_frame
        frame_h, frame_w = frame.shape[:2]

        top_px    = int(frame_h * self.p_roi_top_ratio)
        bottom_px = int(frame_h * self.p_roi_bottom_ratio)
        left_px   = int(frame_w * self.p_roi_left_ratio)
        right_px  = int(frame_w * self.p_roi_right_ratio)

        # Seguridad básica para evitar ROI inválida
        top_px    = max(0, min(top_px, frame_h - 1))
        bottom_px = max(top_px + 1, min(bottom_px, frame_h))
        left_px   = max(0, min(left_px, frame_w - 1))
        right_px  = max(left_px + 1, min(right_px, frame_w))

        roi = frame[top_px:bottom_px, left_px:right_px]

        binary = self._binarize(roi)
        candidates = self._find_stripe_candidates(binary)

        raw_zebra, info = self._validate_zebra(candidates)
        stable_zebra = self._update_debounce(raw_zebra)

        zebra_msg = Bool()
        zebra_msg.data = bool(stable_zebra)
        self.zebra_pub.publish(zebra_msg)

        stripes_msg = Int32()
        stripes_msg.data = int(info['n'])
        self.stripes_pub.publish(stripes_msg)

        if self.p_debug_view:
            vis = frame.copy()

            cv2.rectangle(
                vis,
                (left_px, top_px),
                (right_px, bottom_px),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                vis,
                'ROI zebra',
                (left_px + 10, top_px + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            cnt_color = (0, 255, 0) if raw_zebra else (160, 160, 160)

            for c in candidates:
                x, y, w, h = c['bbox']

                cv2.rectangle(
                    vis,
                    (x + left_px, y + top_px),
                    (x + w + left_px, y + h + top_px),
                    cnt_color,
                    2,
                )

                cx, cy = c['centroid']
                cv2.circle(
                    vis,
                    (int(cx) + left_px, int(cy) + top_px),
                    4,
                    (0, 0, 255),
                    -1,
                )

            estado = 'ZEBRA: YES' if stable_zebra else 'ZEBRA: no'
            estado_color = (0, 255, 0) if stable_zebra else (0, 0, 255)

            cv2.putText(
                vis,
                estado,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                estado_color,
                2,
            )

            cv2.putText(
                vis,
                f"rayas={info['n']}  cv={info['cv']}  "
                f"perp={info['perp_ratio']}  [{info['reason']}]",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"on={self._on_counter}/{self.p_debounce_on}  "
                f"off={self._off_counter}/{self.p_debounce_off}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 255),
                1,
            )

            cv2.putText(
                vis,
                f"roi L={self.p_roi_left_ratio:.2f} R={self.p_roi_right_ratio:.2f}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 255),
                1,
            )

            self._pub_img(self.dbg_vis_pub,  vis,    'bgr8')
            self._pub_img(self.dbg_mask_pub, binary, 'mono8')

def main(args=None):
    rclpy.init(args=args)

    node = ZebraDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()

