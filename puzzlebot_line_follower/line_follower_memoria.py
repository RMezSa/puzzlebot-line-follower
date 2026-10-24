#!/usr/bin/env python3
"""Nodo ROS 2 ligero para detectar linea y publicar /line_error_px.

Usa memoria de ancla para evitar saltos de carril y se desactiva durante
maniobras de cruce mediante /crossing_active.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import Image, CompressedImage

def fresh_qos(depth=1):
    """QoS de mínima latencia: best-effort, solo el frame más nuevo."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.get_logger().info('Line follower node started (solo línea).')

        self.bridge     = CvBridge()
        self.last_frame = None

        self.smoothed_error_px  = 0.0
        self.lost_count         = 0
        self.last_anchor_x      = None
        self.last_anchor_area   = None   # [FIX 3] área del blob ancla actual
        self.anchor_lost_count  = 0
        self._crossing_active   = False
        self._clahe             = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.declare_parameter('compressed_topic',          '/video_source/compressed')
        self.declare_parameter('error_topic',               '/line_error_px')

        self.declare_parameter('roi_top_ratio',             0.80)
        self.declare_parameter('roi_bottom_ratio',          0.90)

        self.declare_parameter('min_area',                  800)
        self.declare_parameter('max_area',                  120000)
        self.declare_parameter('error_alpha',               0.4)
        self.declare_parameter('adaptive_block_size',       51)
        self.declare_parameter('adaptive_c',                7)

        # [FIX 3] valores más conservadores
        self.declare_parameter('anchor_memory_weight',      0.85)   # era 0.6
        self.declare_parameter('anchor_max_jump_px',        50.0)   # era 80
        self.declare_parameter('anchor_reset_after',        20)     # era 10

        # [FIX 3] fracción mínima del área del blob actual para aceptar candidato
        self.declare_parameter('anchor_min_area_ratio',     0.4)

        self.declare_parameter('lookahead_band_px',         5)
        self.declare_parameter('debug_view',                False)

        compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        error_topic      = self.get_parameter('error_topic').get_parameter_value().string_value

        self._cache_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.sub = self.create_subscription(
            CompressedImage, compressed_topic,
            self.camera_callback, fresh_qos(1)
        )
        self.get_logger().info(f'Suscrito a {compressed_topic} (JPEG, fresh).')

        self.create_subscription(Bool, '/crossing_active', self.cb_crossing, 10)

        self.error_pub = self.create_publisher(Float32, error_topic, 10)

        self.dbg_vis_pub   = self.create_publisher(Image, '/debug/line_vis',     1)
        self.dbg_upper_pub = self.create_publisher(Image, '/debug/binary_upper', 1)
        self.dbg_lower_pub = self.create_publisher(Image, '/debug/binary_lower', 1)

        self.timer = self.create_timer(0.05, self.timer_callback)

    def _cache_params(self):
        self.p_roi_top_ratio    = float(self.get_parameter('roi_top_ratio').value)
        self.p_roi_bottom_ratio = float(self.get_parameter('roi_bottom_ratio').value)
        self.p_min_area         = int(self.get_parameter('min_area').value)
        self.p_max_area         = int(self.get_parameter('max_area').value)
        self.p_error_alpha      = float(self.get_parameter('error_alpha').value)
        self.p_debug_view       = bool(self.get_parameter('debug_view').value)

        block = int(self.get_parameter('adaptive_block_size').value)
        self.p_adaptive_block_size = block if block % 2 == 1 else block + 1
        self.p_adaptive_c          = int(self.get_parameter('adaptive_c').value)

        self.p_anchor_memory_weight  = float(self.get_parameter('anchor_memory_weight').value)
        self.p_anchor_max_jump_px    = float(self.get_parameter('anchor_max_jump_px').value)
        self.p_anchor_reset_after    = int(self.get_parameter('anchor_reset_after').value)
        self.p_anchor_min_area_ratio = float(self.get_parameter('anchor_min_area_ratio').value)
        self.p_lookahead_band_px     = int(self.get_parameter('lookahead_band_px').value)

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

    def cb_crossing(self, msg: Bool):
        was_active = self._crossing_active
        self._crossing_active = msg.data

        if was_active and not msg.data:
            self.last_anchor_x     = None
            self.last_anchor_area  = None
            self.anchor_lost_count = 0
            self.smoothed_error_px = 0.0
            self.lost_count        = 0
            self.get_logger().info('Anchor memory reset tras cruce.')

    def _binarize_strip(self, strip):
        gray   = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        gray   = self._clahe.apply(gray)                    
        gray   = cv2.GaussianBlur(gray, (3, 3), 0)          
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,                 
            cv2.THRESH_BINARY_INV,
            blockSize=self.p_adaptive_block_size,
            C=self.p_adaptive_c,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    # Selección de ancla  [FIX 3]
    def _select_anchor(self, blobs_lower, center_x):
        """
        blobs_lower: lista de (cx, cy, pixel_count)

        Lógica anti lane-switch:
        1. Con memoria activa, descarta blobs más pequeños que
           anchor_min_area_ratio * área_actual (artefactos de sobreexposición).
        2. La función de costo pondera fuertemente la posición anterior.
        3. Si el mejor candidato salta más de anchor_max_jump_px, congela
           la posición anterior en lugar de aceptar el salto.
        """
        if not blobs_lower:
            self.anchor_lost_count += 1
            return None

        has_memory = (
            self.last_anchor_x is not None
            and self.anchor_lost_count < self.p_anchor_reset_after
        )

        # Filtrar blobs demasiado pequeños cuando hay memoria
        candidates = blobs_lower
        if has_memory and self.last_anchor_area is not None and self.p_anchor_min_area_ratio > 0.0:
            min_area_threshold = self.last_anchor_area * self.p_anchor_min_area_ratio
            filtered = [b for b in blobs_lower if b[2] >= min_area_threshold]
            if filtered:
                candidates = filtered

        if has_memory:
            w = self.p_anchor_memory_weight
            def cost(b):
                return w * abs(b[0] - self.last_anchor_x) + (1.0 - w) * abs(b[0] - center_x)
        else:
            def cost(b):
                return abs(b[0] - center_x)

        best     = min(candidates, key=cost)
        anchor_x = best[0]
        anchor_a = best[2]

        if has_memory and abs(anchor_x - self.last_anchor_x) > self.p_anchor_max_jump_px:
            # Salto demasiado grande → congelar posición anterior
            self.anchor_lost_count += 1
            return self.last_anchor_x
        else:
            self.anchor_lost_count = 0
            self.last_anchor_x    = anchor_x
            self.last_anchor_area = anchor_a
            return anchor_x

    def _detect_line(self, frame):
        frame_h, frame_w = frame.shape[:2]
        upper_top_px = int(frame_h * self.p_roi_top_ratio)
        lower_top_px = int(frame_h * self.p_roi_bottom_ratio)
        center_x     = frame_w / 2.0

        full_roi    = frame[upper_top_px:, :]
        binary_full = self._binarize_strip(full_roi)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_full, connectivity=4
        )

        valid_labels = set()
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.p_min_area <= area <= self.p_max_area:
                valid_labels.add(i)

        lower_offset = lower_top_px - upper_top_px

        binary_upper = binary_full[:lower_offset, :].copy()
        binary_lower = binary_full[lower_offset:, :].copy()

        blobs_lower  = []
        labels_lower = labels[lower_offset:, :]
        for i in valid_labels:
            ys_lo, xs_lo = np.where(labels_lower == i)
            if len(xs_lo) > 0:
                blobs_lower.append((xs_lo.mean(), ys_lo.mean(), len(xs_lo)))

        anchor_x     = self._select_anchor(blobs_lower, center_x)
        cx_lower     = anchor_x
        anchor_label = None

        if anchor_x is not None:
            anchor_row = lower_offset + (binary_full.shape[0] - lower_offset) // 2
            anchor_col = int(np.clip(anchor_x, 0, frame_w - 1))

            label_at_anchor = int(labels[anchor_row, anchor_col])
            if label_at_anchor in valid_labels:
                anchor_label = label_at_anchor
            else:
                row        = labels[anchor_row, :]
                candidates = np.where(np.isin(row, list(valid_labels)))[0]
                if len(candidates) > 0:
                    nearest      = candidates[np.argmin(np.abs(candidates - anchor_col))]
                    anchor_label = int(row[nearest])

        cx_upper = cy_upper = error_px = None

        if anchor_label is not None:
            labels_upper = labels[:lower_offset, :]
            ys_up, xs_up = np.where(labels_upper == anchor_label)
            if len(xs_up) > 0:
                min_y     = ys_up.min()
                band_mask = ys_up < (min_y + self.p_lookahead_band_px)
                cx_upper  = float(xs_up[band_mask].mean())
                cy_upper  = float(ys_up[band_mask].mean())
                error_px  = cx_upper - center_x
            elif cx_lower is not None:
                error_px = cx_lower - center_x

        return (
            error_px,
            cx_upper, cy_upper,
            cx_lower,
            binary_upper, binary_lower,
            frame_w,
            upper_top_px, lower_top_px,
        )

    def _pub_img(self, publisher, img, encoding='bgr8'):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error publicando imagen de debug: {e}')

    def timer_callback(self):
        if self.last_frame is None:
            return

        if self._crossing_active:
            return

        frame = self.last_frame

        (
            error_px_raw,
            cx_upper, cy_upper,
            cx_lower,
            binary_upper, binary_lower,
            frame_w,
            upper_top_px, lower_top_px,
        ) = self._detect_line(frame)

        out_msg = Float32()
        if error_px_raw is not None:
            self.lost_count = 0
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

        if self.p_debug_view:
            frame_h = frame.shape[0]
            vis     = frame.copy()

            cv2.line(vis, (0, upper_top_px), (frame_w, upper_top_px), (255, 165, 0), 2)
            cv2.line(vis, (0, lower_top_px), (frame_w, lower_top_px), (255, 255, 0), 2)
            cv2.line(vis, (frame_w // 2, upper_top_px), (frame_w // 2, frame_h), (255, 0, 0), 2)

            if cx_lower is not None:
                cy_lower_abs = lower_top_px + int((frame_h - lower_top_px) / 2)
                cv2.circle(vis, (int(cx_lower), cy_lower_abs), 10, (255, 0, 255), -1)
                area_str = f'{int(self.last_anchor_area)}px²' if self.last_anchor_area else '?'
                cv2.putText(vis,
                    f'anchor={cx_lower:.0f} (lost={self.anchor_lost_count}) [{area_str}]',
                    (int(cx_lower) + 12, cy_lower_abs),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2)

            if cx_upper is not None:
                cy_upper_abs = upper_top_px + int(cy_upper)
                cv2.circle(vis, (int(cx_upper), cy_upper_abs), 8, (0, 255, 255), -1)
                cv2.putText(vis,
                    f'err={self.smoothed_error_px:.1f}px (raw={error_px_raw:.1f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, 'TRACKING', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, f'LOST ({self.lost_count})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(vis, 'ROI lookahead',
                (frame_w - 200, upper_top_px + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(vis, 'ROI ancla',
                (frame_w - 160, lower_top_px + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            self._pub_img(self.dbg_vis_pub,   vis,          'bgr8')
            self._pub_img(self.dbg_upper_pub, binary_upper, 'mono8')
            self._pub_img(self.dbg_lower_pub, binary_lower, 'mono8')

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

if __name__ == '__main__':
    main()

