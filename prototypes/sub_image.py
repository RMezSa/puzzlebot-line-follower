#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class SemaforoDetector(Node):
    def __init__(self):
        super().__init__('semaforo_detector')
        self.get_logger().info("Nodo de detección de semáforos iniciado...")
        self.frame_procesado = None
        self.bridge = CvBridge()
        
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.sub = self.create_subscription(
            Image, 
            '/video_source/raw', 
            self.camera_callback, 
            qos_profile=qos
        )
        self.timer = self.create_timer(0.05, self.timer_callback)

    def dibujar_contornos(self, mask, nombre_color, color_bgr, frame):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 1000: 
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
                cv2.putText(frame, nombre_color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)

    def camera_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convertir a espacio HSV
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # --- RANGOS DE COLOR HSV ---
        bajo_verde = np.array([45, 100, 50])
        alto_verde = np.array([90, 255, 255])
        
        bajo_amarillo = np.array([22, 120, 100])
        alto_amarillo = np.array([35, 255, 255])
        
        bajo_rojo1 = np.array([0, 100, 70])
        alto_rojo1 = np.array([10, 255, 255])
        bajo_rojo2 = np.array([165, 120, 70])
        alto_rojo2 = np.array([179, 255, 255])

        # --- CREACIÓN DE MÁSCARAS ---
        mask_verde = cv2.inRange(hsv, bajo_verde, alto_verde)
        mask_amarillo = cv2.inRange(hsv, bajo_amarillo, alto_amarillo)
        
        mask_rojo1 = cv2.inRange(hsv, bajo_rojo1, alto_rojo1)
        mask_rojo2 = cv2.inRange(hsv, bajo_rojo2, alto_rojo2)
        mask_rojo = cv2.add(mask_rojo1, mask_rojo2)
        self.dibujar_contornos(mask_verde, "Verde", (0, 255, 0), frame)
        self.dibujar_contornos(mask_amarillo, "Amarillo", (0, 255, 255), frame)
        self.dibujar_contornos(mask_rojo, "Rojo", (0, 0, 255), frame)
        # Guardar el frame procesado para que el timer lo muestre
        self.frame_procesado = frame

    def timer_callback(self):
        if self.frame_procesado is not None:
            cv2.imshow("Deteccion de Semaforos ROS 2", self.frame_procesado)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SemaforoDetector()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        print("Terminado por el usuario")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()