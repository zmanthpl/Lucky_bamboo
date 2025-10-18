import cv2
import numpy as np
import json
import time
import torch
import socket
import threading
import struct
import queue
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox


# ===== 自定义 JSON 编码器 =====
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


# ===== 配置参数 =====
# 视频流输入配置
TCP_INPUT_IP = "0.0.0.0"  # 监听所有接口
TCP_INPUT_PORT = 6006  # 视频流输入端口

# 视频流输出配置
TCP_OUTPUT_IP = "0.0.0.0"  # 服务器IP，0.0.0.0表示监听所有接口
TCP_OUTPUT_PORT = 6008  # 视频流输出端口

# 检测结果输出配置
RESULT_OUTPUT_PORT = 6009  # 新增：检测结果输出端口

MODEL_PATH = "last/yolov7_se_Wiou/best.pt"
IMG_SIZE = 1280
CONF_THRESH = 0.30
IOU_THRESH = 0.40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 全局变量 =====
stop_threads = False
frame_queue = queue.Queue(maxsize=5)  # 用于存储接收到的帧
output_clients = []  # 存储所有连接的输出客户端
output_clients_lock = threading.Lock()  # 保护输出客户端列表
result_clients = []  # 存储所有连接的结果客户端
result_clients_lock = threading.Lock()  # 保护结果客户端列表

# ===== 加载模型 =====
print("加载YOLOv7模型...")
model = attempt_load(MODEL_PATH, map_location=DEVICE)
model.eval()
if DEVICE != "cpu":
    model.half()
    print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
print("模型加载完成!")


# ===== TCP服务器函数 =====
def tcp_input_server():
    """作为服务器接收树莓派发送的视频流"""
    global stop_threads

    server_socket = None
    try:
        # 创建TCP服务器套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((TCP_INPUT_IP, TCP_INPUT_PORT))
        server_socket.listen(5)
        server_socket.settimeout(1.0)

        print(f"视频输入服务器已启动，监听 {TCP_INPUT_IP}:{TCP_INPUT_PORT}")
        print("等待树莓派连接...")

        while not stop_threads:
            try:
                # 接受新连接
                client_socket, client_address = server_socket.accept()
                print(f"树莓派已连接: {client_address}")

                # 为每个客户端创建处理线程
                client_thread = threading.Thread(
                    target=handle_input_client,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()

            except socket.timeout:
                continue
            except Exception as e:
                print(f"接受客户端连接时出错: {e}")

    except Exception as e:
        print(f"输入服务器错误: {e}")
    finally:
        if server_socket:
            server_socket.close()
            server_socket = None

    print("TCP输入服务器线程退出")


def handle_input_client(client_socket):
    """处理输入客户端连接"""
    global stop_threads

    client_socket.settimeout(1.0)

    try:
        while not stop_threads:
            # 确保读取完整的4字节帧头
            header = b''
            while len(header) < 4 and not stop_threads:
                try:
                    chunk = client_socket.recv(4 - len(header))
                    if not chunk:
                        raise ConnectionError("连接中断")
                    header += chunk
                except socket.timeout:
                    continue

            if len(header) < 4 or stop_threads:
                break

            # 解析帧长度
            img_size = struct.unpack('!I', header)[0]

            # 接收图像数据
            img_data = b''
            while len(img_data) < img_size and not stop_threads:
                try:
                    to_read = min(4096, img_size - len(img_data))
                    packet = client_socket.recv(to_read)
                    if not packet:
                        raise ConnectionError("连接中断")
                    img_data += packet
                except socket.timeout:
                    continue

            if len(img_data) != img_size or stop_threads:
                continue

            # 尝试解码图像
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                # 将帧放入队列
                try:
                    frame_queue.put(img, timeout=0.1)
                except queue.Full:
                    pass  # 如果队列满，跳过此帧
            else:
                print("解码失败! 接收数据长度:", len(img_data))

    except ConnectionError as e:
        print(f"连接错误: {e}")
    except socket.error as e:
        print(f"套接字错误: {e}")
    except Exception as e:
        print(f"接收视频时出错: {e}")
    finally:
        # 清理资源
        client_socket.close()
        print("输入客户端连接已关闭")


def tcp_output_server():
    """作为服务器向本地电脑发送处理后的视频流"""
    global stop_threads, output_clients

    server_socket = None
    try:
        # 创建TCP服务器套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((TCP_OUTPUT_IP, TCP_OUTPUT_PORT))
        server_socket.listen(5)
        server_socket.settimeout(1.0)

        print(f"视频输出服务器已启动，监听 {TCP_OUTPUT_IP}:{TCP_OUTPUT_PORT}")
        print("等待本地电脑连接...")

        while not stop_threads:
            try:
                # 接受新连接
                client_socket, client_address = server_socket.accept()
                print(f"新的输出客户端连接: {client_address}")

                with output_clients_lock:
                    # 关闭旧的连接（如果有）
                    for old_client in output_clients:
                        try:
                            old_client.close()
                        except:
                            pass
                    # 添加新连接
                    output_clients = [client_socket]

            except socket.timeout:
                continue
            except Exception as e:
                print(f"接受客户端连接时出错: {e}")

            # 定期清理断开连接的客户端
            with output_clients_lock:
                active_clients = []
                for client in output_clients:
                    try:
                        # 尝试发送一个空数据来检查连接状态
                        client.send(b'')
                        active_clients.append(client)
                    except:
                        try:
                            client.close()
                        except:
                            pass
                output_clients = active_clients

    except Exception as e:
        print(f"输出服务器错误: {e}")
    finally:
        # 清理资源
        with output_clients_lock:
            for client in output_clients:
                try:
                    client.close()
                except:
                    pass
            output_clients = []

        if server_socket:
            server_socket.close()
            server_socket = None

    print("TCP输出服务器线程退出")


def tcp_result_server():
    """作为服务器向树莓派发送检测结果"""
    global stop_threads, result_clients

    server_socket = None
    try:
        # 创建TCP服务器套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", RESULT_OUTPUT_PORT))
        server_socket.listen(5)
        server_socket.settimeout(1.0)

        print(f"结果输出服务器已启动，监听 0.0.0.0:{RESULT_OUTPUT_PORT}")
        print("等待树莓派连接...")

        while not stop_threads:
            try:
                # 接受新连接
                client_socket, client_address = server_socket.accept()
                print(f"新的结果客户端连接: {client_address}")

                with result_clients_lock:
                    # 关闭旧的连接（如果有）
                    for old_client in result_clients:
                        try:
                            old_client.close()
                        except:
                            pass
                    # 添加新连接
                    result_clients = [client_socket]

            except socket.timeout:
                continue
            except Exception as e:
                print(f"接受结果客户端连接时出错: {e}")

            # 定期清理断开连接的客户端
            with result_clients_lock:
                active_clients = []
                for client in result_clients:
                    try:
                        # 尝试发送一个空数据来检查连接状态
                        client.send(b'')
                        active_clients.append(client)
                    except:
                        try:
                            client.close()
                        except:
                            pass
                result_clients = active_clients

    except Exception as e:
        print(f"结果服务器错误: {e}")
    finally:
        # 清理资源
        with result_clients_lock:
            for client in result_clients:
                try:
                    client.close()
                except:
                    pass
            result_clients = []

        if server_socket:
            server_socket.close()
            server_socket = None

    print("TCP结果服务器线程退出")


def send_frame_to_clients(frame):
    """将帧发送给所有连接的输出客户端（本地电脑）"""
    global output_clients

    if frame is None:
        return

    # 编码帧为JPEG格式
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 80%质量
    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

    if not result:
        return

    # 转换为字节
    frame_data = encoded_frame.tobytes()

    # 添加帧头（帧长度）
    header = struct.pack('!I', len(frame_data))
    message = header + frame_data

    with output_clients_lock:
        active_clients = []
        for client in output_clients:
            try:
                client.sendall(message)
                active_clients.append(client)
            except Exception as e:
                print(f"发送帧到客户端失败: {e}")
                try:
                    client.close()
                except:
                    pass
        output_clients = active_clients


def send_result_to_clients(result_data):
    """将检测结果发送给所有连接的结果客户端（树莓派）"""
    global result_clients

    if result_data is None:
        return

    # 将结果转换为JSON格式
    try:
        result_json = json.dumps(result_data, cls=NumpyEncoder)
        result_bytes = result_json.encode('utf-8')

        # 添加数据头（数据长度）
        header = struct.pack('!I', len(result_bytes))
        message = header + result_bytes

        with result_clients_lock:
            active_clients = []
            for client in result_clients:
                try:
                    client.sendall(message)
                    active_clients.append(client)
                except Exception as e:
                    print(f"发送结果到客户端失败: {e}")
                    try:
                        client.close()
                    except:
                        pass
            result_clients = active_clients

    except Exception as e:
        print(f"编码结果数据失败: {e}")


def processing_thread():
    """处理图像帧的线程"""
    global stop_threads

    while not stop_threads:
        try:
            # 从队列获取帧
            img = frame_queue.get(timeout=0.5)

            if img is not None:
                # 处理图像
                start_time = time.time()
                # 执行目标检测
                results, proc_time = detect_objects(img)

                # 发送检测结果到树莓派
                result_data = {
                    "detections": results,
                }
                send_result_to_clients(result_data)

                # 绘制检测结果
                display_frame = draw_detections(img, results)

                # 计算并显示FPS
                fps = 1.0 / (time.time() - start_time)
                #cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"检测完成: {len(results)} 个目标, 耗时: {proc_time:.3f}s, FPS: {fps:.1f}")

                # 发送处理后的帧到所有客户端（本地电脑）
                send_frame_to_clients(display_frame)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"处理图像时出错: {e}")

    print("处理线程退出")


def detect_objects(img):
    """使用YOLOv7检测图像中的竹节并计算切线"""
    start_detect = time.time()
    img0 = img.copy()
    # 预处理
    img = letterbox(img, IMG_SIZE, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE)
    img = img.half() if DEVICE != "cpu" else img.float()
    img = img / 255.0
    img = img.unsqueeze(0)
    # 推理
    with torch.no_grad():
        pred = model(img)[0]
    # 后处理
    pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)
    # 处理结果
    detections = []
    if pred[0] is not None:
        # 缩放坐标
        pred_boxes = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
        for i, det in enumerate(pred[0]):
            # 获取坐标
            xyxy = pred_boxes[i]
            x1, y1, x2, y2 = map(int, xyxy)
            # 获取置信度
            conf = det[4].item()
            # ===== 计算竹节切线端点 =====
            # 计算竹节中心高度
            center_y = (y1 + y2) // 2
            # 左侧切线端点 (从顶部到底部)
            left_tangent_top = [x1, y1]
            left_tangent_bottom = [x1, y2]
            # 右侧切线端点 (从顶部到底部)
            right_tangent_top = [x2, y1]
            right_tangent_bottom = [x2, y2]
            # 添加到结果
            detections.append({
                # 切线端点坐标
                "center_point_left": [(left_tangent_top[0] + left_tangent_bottom[0]) // 2,
                                      (left_tangent_top[1] + left_tangent_bottom[1]) // 2],
                "center_point_right": [(right_tangent_top[0] + right_tangent_bottom[0]) // 2,
                                       (right_tangent_top[1] + right_tangent_bottom[1]) // 2],
                "center_y": center_y,  # 中心高度，可用于排序
                "conf": conf,
            })
    return detections, time.time() - start_detect


def draw_detections(frame, detections):
    """在图像上绘制检测结果和切线"""
    display_frame = frame.copy()
    # 绘制每个检测到的竹节
    for detection in detections:
        # ===== 绘制切线 =====
        # 获取切线端点
        left_center = detection["center_point_left"]
        right_center = detection["center_point_right"]
        # 绘制切线 (红色)
        cv2.line(display_frame, tuple(left_center), tuple(right_center), (0, 0, 255), 2)
        cv2.circle(display_frame, tuple(left_center), 5, (255, 0, 0), -1)
        cv2.circle(display_frame, tuple(right_center), 5, (255, 0, 0), -1)
        conf = detection["conf"]
        # 绘制置信度
        label = f"{conf:.2f}"
        cv2.putText(display_frame, label, (right_center[0], left_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # 显示检测到的竹节数量
    count_label = f"{len(detections)}"
    cv2.putText(display_frame, count_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 添加时间戳
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display_frame, timestamp, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return display_frame


# ===== 主程序 =====
if __name__ == "__main__":
    # 启动TCP输入服务器线程（接收树莓派视频）
    tcp_input_thread = threading.Thread(target=tcp_input_server, daemon=True)
    tcp_input_thread.start()

    # 启动TCP输出服务器线程（发送处理后的视频到本地电脑）
    tcp_output_thread = threading.Thread(target=tcp_output_server, daemon=True)
    tcp_output_thread.start()

    # 启动TCP结果服务器线程（发送检测结果到树莓派）
    tcp_result_thread = threading.Thread(target=tcp_result_server, daemon=True)
    tcp_result_thread.start()

    # 启动处理线程
    process_thread = threading.Thread(target=processing_thread, daemon=True)
    process_thread.start()

    print("服务器已启动，等待连接...")
    print(f"树莓派可以连接视频输入端口: {TCP_INPUT_PORT}")
    print(f"树莓派可以连接结果输出端口: {RESULT_OUTPUT_PORT}")
    print(f"本地电脑可以连接视频输出端口: {TCP_OUTPUT_PORT}")

    try:
        # 服务器主循环
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("程序终止")
    finally:
        stop_threads = True
        tcp_input_thread.join(timeout=1.0)
        tcp_output_thread.join(timeout=1.0)
        tcp_result_thread.join(timeout=1.0)
        process_thread.join(timeout=1.0)
        print("资源已释放")