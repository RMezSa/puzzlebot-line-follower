#!/usr/bin/env python3
"""
SignDetector — ROS 2 Humble
Jetson Orin Nano · /video_source/compressed (sensor_msgs/CompressedImage)

Nodo PESADO: SOLO inferencia YOLOv8.
Corre en su propio proceso para que NO bloquee al line_follower.
Ambos nodos se suscriben al mismo /video_source/compressed (best-effort).

Señales del modelo:
  stop, give_way, flecha_derecha, flecha_izquierda,
  flecha_derecho, zona_obras, glorieta
Semáforo (clases propias del modelo):
  red, yellow, green

Publica:
  /sign_detections  (std_msgs/String)  — lista JSON de detecciones
  /semaforo         (std_msgs/Int32)   — red=1 yellow=2 green=3 / ninguno=0

Debug topics (si debug_view:=true):
  /debug/sign_vis   — frame con bounding boxes de las detecciones

NOTA sobre el flip:
  El flip horizontal lo aplica el nodo de cámara (cpp_camera, Hflip=true)
  antes de publicar. Todos los nodos de visión reciben la imagen ya corregida;
  no hace falta flipear aquí ni usar LABEL_FIX.
"""

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
from std_msgs.msg import Int32, String
from sensor_msgs.msg import Image, CompressedImage


def fresh_qos(depth=1):
    """QoS de mínima latencia: best-effort, solo el frame más nuevo."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


# ── YOLOv8 — importación diferida para no crashear si no está instalado ───────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Mapping de clase YOLO → entero de semáforo
SIGN_TO_SEMAFORO = {
    'red':    1,
    'yellow': 2,
    'green':  3,
}


class SignDetector(Node):
    def __init__(self):
        super().__init__('sign_detector')
        self.get_logger().info('Sign detector node started (solo YOLOv8).')

        self.bridge     = CvBridge()
        self.last_frame = None

        self.yolo_model      = None
        self.last_detections = []

        # ── Parámetros ───────────────────────────────────────────────────────
        self.declare_parameter('compressed_topic', '/video_source/compressed')
        self.declare_parameter('sign_topic',    '/sign_detections')
        self.declare_parameter('semaforo_topic','/semaforo')

        self.declare_parameter('yolo_model_path',
            '/home/puzzlebot/ros2_ws/src/my_opencv_demo/my_opencv_demo/best3.pt')
        self.declare_parameter('yolo_conf',     0.45)
        self.declare_parameter('yolo_iou',      0.45)
        self.declare_parameter('yolo_imgsz',    320)
        self.declare_parameter('yolo_device',   '')
        self.declare_parameter('yolo_enabled',  True)

        # Periodo del timer de inferencia. El modelo corre a su propio ritmo;
        # si tarda más que esto, simplemente corre lo más rápido que puede.
        self.declare_parameter('infer_period',  0.10)   # ~10 Hz objetivo

        self.declare_parameter('debug_view',    False)

        # ── Leer topics ──────────────────────────────────────────────────────
        compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        sign_topic       = self.get_parameter('sign_topic').get_parameter_value().string_value
        semaforo_topic   = self.get_parameter('semaforo_topic').get_parameter_value().string_value

        self._cache_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        # ── Subscripción a la cámara (depth=1 → siempre el frame más nuevo) ──
        self.sub = self.create_subscription(
            CompressedImage, compressed_topic,
            self.camera_callback, fresh_qos(1)
        )
        self.get_logger().info(f'Suscrito a {compressed_topic} (JPEG, fresh).')

        # ── Publicadores ─────────────────────────────────────────────────────
        self.sign_pub = self.create_publisher(String, sign_topic,     10)
        self.sem_pub  = self.create_publisher(Int32,  semaforo_topic, 10)
        self.dbg_pub  = self.create_publisher(Image,  '/debug/sign_vis', 1)

        infer_period = float(self.get_parameter('infer_period').value)
        self.timer   = self.create_timer(infer_period, self.timer_callback)

        # ── Cargar modelo ────────────────────────────────────────────────────
        self._load_yolo()

    # ─────────────────────────────────────────────────────────────────────────
    # Carga del modelo
    # ─────────────────────────────────────────────────────────────────────────
    def _load_yolo(self):
        if not self.p_yolo_enabled:
            self.get_logger().info('YOLO deshabilitado por parámetro.')
            return
        if not YOLO_AVAILABLE:
            self.get_logger().error(
                'ultralytics no está instalado. Instálalo con:  pip install ultralytics'
            )
            return

        model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        device     = self.get_parameter('yolo_device').get_parameter_value().string_value or None

        try:
            t0 = time.time()
            self.yolo_model = YOLO(model_path)
            dummy = np.zeros((self.p_yolo_imgsz, self.p_yolo_imgsz, 3), dtype=np.uint8)
            self.yolo_model.predict(
                dummy,
                conf=self.p_yolo_conf,
                iou=self.p_yolo_iou,
                imgsz=self.p_yolo_imgsz,
                device=device,
                verbose=False,
            )
            self.get_logger().info(
                f'Modelo YOLO cargado desde "{model_path}" '
                f'(device={device}, warm-up en {time.time() - t0:.2f}s)'
            )
        except Exception as e:
            self.yolo_model = None
            self.get_logger().error(f'No se pudo cargar el modelo YOLO: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Caché de parámetros
    # ─────────────────────────────────────────────────────────────────────────
    def _cache_params(self):
        self.p_yolo_enabled = bool(self.get_parameter('yolo_enabled').value)
        self.p_yolo_conf    = float(self.get_parameter('yolo_conf').value)
        self.p_yolo_iou     = float(self.get_parameter('yolo_iou').value)
        self.p_yolo_imgsz   = int(self.get_parameter('yolo_imgsz').value)
        self.p_debug_view   = bool(self.get_parameter('debug_view').value)

    def _on_params_changed(self, params):
        prev_enabled    = self.p_yolo_enabled
        prev_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self._cache_params()
        new_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        if self.p_yolo_enabled and (not prev_enabled or new_model_path != prev_model_path):
            self._load_yolo()
        return SetParametersResult(successful=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Camera callback
    # ─────────────────────────────────────────────────────────────────────────
    def camera_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('imdecode devolvió None.')
            return
        self.last_frame = frame

    # ─────────────────────────────────────────────────────────────────────────
    # Inferencia YOLO
    # ─────────────────────────────────────────────────────────────────────────
    def _run_yolo(self, frame):
        if not self.p_yolo_enabled or self.yolo_model is None:
            return []

        device = self.get_parameter('yolo_device').get_parameter_value().string_value or None

        try:
            results = self.yolo_model.predict(
                frame,
                conf=self.p_yolo_conf,
                iou=self.p_yolo_iou,
                imgsz=self.p_yolo_imgsz,
                device=device,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference error: {e}')
            return []

        if not results:
            return []

        result = results[0]
        names  = result.names

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            area       = (x2 - x1) * (y2 - y1)
            class_name = names.get(cls_id, str(cls_id))


            if class_name == 'flecha_derecha':
                class_name = 'flecha_izquierda'
            elif class_name == 'flecha_izquierda':
                class_name = 'flecha_derecha'
            detections.append({
                'class_id':   cls_id,
                'class_name': class_name,
                'confidence': round(conf, 3),
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'cx': (x1 + x2) // 2,
                'cy': (y1 + y2) // 2,
                'bbox_area':  area,
                'is_closest': False,
            })

        if detections:
            closest_idx = max(range(len(detections)),
                              key=lambda i: detections[i]['bbox_area'])
            detections[closest_idx]['is_closest'] = True

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # Semáforo desde YOLO
    # ─────────────────────────────────────────────────────────────────────────
    def _semaforo_from_detections(self, detections):
        for det in sorted(detections, key=lambda d: d['bbox_area'], reverse=True):
            sem_val = SIGN_TO_SEMAFORO.get(det['class_name'])
            if sem_val is not None:
                return sem_val
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay de detecciones
    # ─────────────────────────────────────────────────────────────────────────
    _SIGN_COLORS = {
        'stop':             (0,   0,   220),
        'give_way':         (0,   165, 255),
        'flecha_derecha':   (255, 200,   0),
        'flecha_izquierda': (255, 200,   0),
        'flecha_derecho':   (255, 200,   0),
        'zona_obras':       (0,   128, 255),
        'glorieta':         (180,   0, 255),
        'red':              (0,   0,   255),
        'yellow':           (0,   220, 255),
        'green':            (0,   255,   0),
    }

    def _draw_yolo(self, vis, detections):
        for det in detections:
            color      = self._SIGN_COLORS.get(det['class_name'], (200, 200, 200))
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            is_closest = det.get('is_closest', False)

            thickness = 4 if is_closest else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            if is_closest:
                cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

            area  = det.get('bbox_area', (x2 - x1) * (y2 - y1))
            label = (
                f"{det['class_name']} {det['confidence']:.2f} "
                f"[{area}px²{'  CLOSEST' if is_closest else ''}]"
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(vis, label, (x1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    def _pub_img(self, publisher, img, encoding='bgr8'):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error publicando imagen de debug: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Timer principal — solo YOLO
    # ─────────────────────────────────────────────────────────────────────────
    def timer_callback(self):
        if self.last_frame is None:
            return

        frame = self.last_frame

        # Inferencia
        self.last_detections = self._run_yolo(frame)

        # Publicar detecciones
        sign_msg = String()
        sign_msg.data = json.dumps(self.last_detections)
        self.sign_pub.publish(sign_msg)

        # Publicar semáforo
        sem_msg = Int32()
        sem_msg.data = self._semaforo_from_detections(self.last_detections)
        self.sem_pub.publish(sem_msg)

        # Debug
        if self.p_debug_view:
            vis = frame.copy()
            self._draw_yolo(vis, self.last_detections)

            n = len(self.last_detections)
            LABEL_MAP = {0: 'NONE', 1: 'RED', 2: 'YELLOW', 3: 'GREEN'}
            TL_COLORS = {0: (200, 200, 200), 1: (0, 0, 255), 2: (0, 220, 255), 3: (0, 255, 0)}
            sem_val = sem_msg.data

            cv2.putText(vis, f'YOLO: {n} det',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(vis, f'Semaforo: {LABEL_MAP[sem_val]}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TL_COLORS[sem_val], 2)

            self._pub_img(self.dbg_pub, vis, 'bgr8')


def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()