from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


def fresh_qos(depth=1):
    """QoS de mínima latencia: best-effort, solo el frame más nuevo."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def pub_img(node, publisher, img, encoding='bgr8'):
    try:
        msg = node.bridge.cv2_to_imgmsg(img, encoding=encoding)
        publisher.publish(msg)
    except Exception as e:
        node.get_logger().warn(f'Error publicando imagen de debug: {e}')
