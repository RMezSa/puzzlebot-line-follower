#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
#

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.get_logger().info('Line follower node started (line + traffic light).')

        self.bridge     = CvBridge()
        self.last_frame = None

        # ── Estado ──────────────────────────────────────────────────────────
        self.smoothed_error_px = 0.0
        self.lost_count        = 0

        # ── HSV ranges para semáforo ────────────────────────────────────────
        self.lower_red1   = np.array([0,   120,  70])
        self.upper_red1   = np.array([10,  255, 255])
        self.lower_red2   = np.array([170, 120,  70])
        self.upper_red2   = np.array([180, 255, 255])
        self.lower_yellow = np.array([18,   50,  80])
        self.upper_yellow = np.array([38,  255, 255])
        self.lower_green  = np.array([39,  100,  80])
        self.upper_green  = np.array([85,  255, 255])

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('image_topic',               '/video_source/raw')
        self.declare_parameter('error_topic',               '/line_error_px')
        self.declare_parameter('semaforo_topic',            '/semaforo')

        # Línea
        self.declare_parameter('roi_top_ratio',             0.5)
        self.declare_parameter('trapezoid_top_width_ratio', 0.9)
        self.declare_parameter('min_area',                  800)
        self.declare_parameter('max_area',                  120000)
        self.declare_parameter('error_alpha',               0.4)
        self.declare_parameter('adaptive_block_size',       51)   # debe ser impar
        self.declare_parameter('adaptive_c',                10)

        # Semáforo
        self.declare_parameter('color_min_area_ratio',      0.01) # fracción del frame total
        self.declare_parameter('yg_ambiguity_ratio',        0.25) # umbral árbitro yellow/green

        # Debug
        self.declare_parameter('debug_view',                False)

        image_topic    = self.get_parameter('image_topic').get_parameter_value().string_value
        error_topic    = self.get_parameter('error_topic').get_parameter_value().string_value
        semaforo_topic = self.get_parameter('semaforo_topic').get_parameter_value().string_value

        self._cache_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.sub        = self.create_subscription(Image, image_topic, self.camera_callback, qos_profile=qos)
        self.error_pub  = self.create_publisher(Float32, error_topic,    10)
        self.sem_pub    = self.create_publisher(Int32,   semaforo_topic, 10)
        self.timer      = self.create_timer(0.05, self.timer_callback)

    # ── Caché ────────────────────────────────────────────────────────────────
    def _cache_params(self):
        self.p_roi_top_ratio             = float(self.get_parameter('roi_top_ratio').value)
        self.p_trapezoid_top_width_ratio = float(self.get_parameter('trapezoid_top_width_ratio').value)
        self.p_min_area                  = int(self.get_parameter('min_area').value)
        self.p_max_area                  = int(self.get_parameter('max_area').value)
        self.p_error_alpha               = float(self.get_parameter('error_alpha').value)
        self.p_debug_view                = bool(self.get_parameter('debug_view').value)
        self.p_color_min_area_ratio      = float(self.get_parameter('color_min_area_ratio').value)
        self.p_yg_ambiguity_ratio        = float(self.get_parameter('yg_ambiguity_ratio').value)

        # adaptive_block_size debe ser impar >= 3
        block = int(self.get_parameter('adaptive_block_size').value)
        self.p_adaptive_block_size = block if block % 2 == 1 else block + 1
        self.p_adaptive_c          = int(self.get_parameter('adaptive_c').value)

    def _on_params_changed(self, params):
        self._cache_params()
        return rclpy.node.SetParametersResult(successful=True)

    # ── Camera callback ──────────────────────────────────────────────────────
    def camera_callback(self, msg):
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')

    # ── Detección de semáforo ────────────────────────────────────────────────
    def _detect_color(self, frame):
        """
        Devuelve int: 0=sin color, 1=rojo, 2=amarillo, 3=verde
        Usa frame completo (el semáforo puede aparecer en cualquier parte).
        Incluye árbitro yellow/green por mediana de Hue.
        """
        COLOR_MAP = {None: 0, 'red': 1, 'yellow': 2, 'green': 3}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_red = (
            cv2.inRange(hsv, self.lower_red1, self.upper_red1) |
            cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        )
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_green  = cv2.inRange(hsv, self.lower_green,  self.upper_green)

        # Morfología para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for mask in (mask_red, mask_yellow, mask_green):
            cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, dst=mask)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)

        counts = {
            'red':    int(cv2.countNonZero(mask_red)),
            'yellow': int(cv2.countNonZero(mask_yellow)),
            'green':  int(cv2.countNonZero(mask_green)),
        }

        best_color = max(counts, key=counts.get)
        best_count = counts[best_color]

        # Threshold dinámico: al menos 1% del frame total
        min_pixels = frame.shape[0] * frame.shape[1] * self.p_color_min_area_ratio
        if best_count < min_pixels:
            return COLOR_MAP[None], counts

        # ── Árbitro yellow/green ─────────────────────────────────────────────
        # Si ambos colores compiten (ratio > umbral), usar mediana del canal Hue
        if best_color in ('yellow', 'green'):
            y_count = counts['yellow']
            g_count = counts['green']
            total   = y_count + g_count
            if total > 0:
                ratio = min(y_count, g_count) / max(y_count, g_count)
                if ratio > self.p_yg_ambiguity_ratio:
                    combined_mask = cv2.bitwise_or(mask_yellow, mask_green)
                    hue_channel   = hsv[:, :, 0]
                    pixels        = hue_channel[combined_mask > 0]
                    if len(pixels) > 0:
                        median_hue = float(np.median(pixels))
                        best_color = 'yellow' if median_hue < 38 else 'green'

        return COLOR_MAP[best_color], counts

    # ── Detección de línea ───────────────────────────────────────────────────
    def _detect_line(self, frame):
        """
        Devuelve (error_px | None, roi, binary, cx, roi_w).
        Usa adaptive threshold para robustez ante cambios de iluminación.
        Selecciona el blob más grande en rango de área válido.
        """
        frame_h, frame_w = frame.shape[:2]

        # ROI
        roi_top = int(frame_h * self.p_roi_top_ratio)
        roi     = frame[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]

        # Máscara trapezoidal
        top_width = int(roi_w * self.p_trapezoid_top_width_ratio)
        trapezoid = np.array([[
            ((roi_w - top_width) // 2, 0),
            ((roi_w + top_width) // 2, 0),
            (roi_w, roi_h),
            (0,     roi_h)
        ]], dtype=np.int32)
        mask      = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(mask, trapezoid, 255)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        # Adaptive threshold — robusto ante iluminación variable
        gray   = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.p_adaptive_block_size,
            C=self.p_adaptive_c
        )

        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Blob principal por ConnectedComponents (más robusto que moments si hay ruido)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
        best_idx  = -1
        best_area =  0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.p_min_area <= area <= self.p_max_area and area > best_area:
                best_area = area
                best_idx  = i

        if best_idx != -1:
            cx, cy = centroids[best_idx]
            error_px = cx - (roi_w / 2.0)
            return error_px, roi, binary, cx, cy, roi_w, roi_h
        else:
            return None, roi, binary, None, None, roi_w, roi_h

    # ── Timer principal ──────────────────────────────────────────────────────
    def timer_callback(self):
        if self.last_frame is None:
            return

        frame = self.last_frame

        # ── 1. Semáforo ──────────────────────────────────────────────────────
        color_int, counts = self._detect_color(frame)
        sem_msg      = Int32()
        sem_msg.data = color_int
        self.sem_pub.publish(sem_msg)

        # ── 2. Línea ─────────────────────────────────────────────────────────
        error_px_raw, roi, binary, cx, cy, roi_w, roi_h = self._detect_line(frame)
        out_msg = Float32()

        if error_px_raw is not None:
            self.lost_count = 0

            # Low-pass filter
            self.smoothed_error_px = (
                self.p_error_alpha * error_px_raw
                + (1.0 - self.p_error_alpha) * self.smoothed_error_px
            )
            out_msg.data = float(self.smoothed_error_px)
            self.error_pub.publish(out_msg)
        else:
            self.lost_count += 1
            out_msg.data = float('nan')
            self.error_pub.publish(out_msg)
            self.get_logger().warn(f'Linea perdida ({self.lost_count}): publicando NaN.')

        # ── 3. Debug ─────────────────────────────────────────────────────────
        if self.p_debug_view:
            LABEL_MAP = {0: 'NONE', 1: 'RED', 2: 'YELLOW', 3: 'GREEN'}
            TL_COLORS = {0: (200, 200, 200), 1: (0, 0, 255), 2: (0, 220, 255), 3: (0, 255, 0)}

            vis = roi.copy()

            # Centro de referencia
            cv2.line(vis, (roi_w // 2, roi_h), (roi_w // 2, 0), (255, 0, 0), 2)

            if cx is not None:
                cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)
                cv2.putText(vis,
                    f'err={self.smoothed_error_px:.1f}px (raw={error_px_raw:.1f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, 'TRACKING', (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f'LOST ({self.lost_count})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(vis,
                f"Semaforo: {LABEL_MAP[color_int]}  "
                f"R:{counts['red']} Y:{counts['yellow']} G:{counts['green']}",
                (10, roi_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                TL_COLORS[color_int], 2)

            cv2.imshow('Line Follower Debug', vis)
            cv2.imshow('Binary Mask', binary)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    