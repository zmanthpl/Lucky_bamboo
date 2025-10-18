import cv2
import numpy as np
import socket
import struct
import time
import threading

# 配置参数
SERVER_IP = "127.0.0.1"  # 服务器地址
SERVER_PORT = 6008  # 视频流输出端口

# 全局变量
stop_receiving = False
reconnect_attempts = 0
max_reconnect_attempts = 10
reconnect_delay = 3  # 重连延迟(秒)


def receive_and_display_video():
    """接收并显示视频流"""
    global stop_receiving, reconnect_attempts

    client_socket = None
    frame_count = 0
    start_time = time.time()
    buffer = b''  # 添加缓冲区以处理不完整的数据

    try:
        # 创建TCP客户端套接字
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5.0)  # 设置5秒超时

        print(f"尝试连接到服务器 {SERVER_IP}:{SERVER_PORT}")
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"已连接到服务器 {SERVER_IP}:{SERVER_PORT}")

        # 重置重连尝试次数
        reconnect_attempts = 0

        # 创建显示窗口
        cv2.namedWindow("Bamboo Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bamboo Detection", 800, 600)

        last_frame_time = time.time()
        fps = 0

        while not stop_receiving:
            try:
                # 确保读取完整的4字节帧头
                while len(buffer) < 4 and not stop_receiving:
                    try:
                        data = client_socket.recv(4096)
                        if not data:
                            print("连接中断：未收到数据")
                            raise ConnectionError("连接中断")
                        buffer += data
                    except socket.timeout:
                        # 超时但未收到数据，继续尝试
                        continue

                if stop_receiving:
                    break

                # 解析帧长度
                frame_size = struct.unpack('!I', buffer[:4])[0]
                buffer = buffer[4:]  # 移除已处理的帧头

                # 验证帧大小
                if frame_size <= 0 or frame_size > 5000000:  # 限制最大帧大小为5MB
                    print(f"异常帧大小: {frame_size}，跳过并清空缓冲区")
                    buffer = b''  # 清空缓冲区
                    continue

                # 读取帧数据
                while len(buffer) < frame_size and not stop_receiving:
                    try:
                        to_read = min(4096, frame_size - len(buffer))
                        data = client_socket.recv(to_read)
                        if not data:
                            print("连接中断：未收到完整帧数据")
                            raise ConnectionError("连接中断")
                        buffer += data
                    except socket.timeout:
                        # 超时但未收到完整数据，继续尝试
                        continue

                if stop_receiving:
                    break

                # 提取完整帧数据
                frame_data = buffer[:frame_size]
                buffer = buffer[frame_size:]  # 移除已处理的帧数据

                # 解码帧
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                frame_count += 1
                if frame is not None:
                    # 计算帧率
                    current_time = time.time()
                    time_diff = current_time - last_frame_time
                    last_frame_time = current_time

                    if time_diff > 0:
                        fps = 0.9 * fps + 0.1 * (1.0 / time_diff)  # 平滑的FPS计算

                    # 在图像上显示FPS
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 显示帧
                    cv2.imshow("Bamboo Detection", frame)

                    # 每30帧打印一次状态
                    if frame_count % 30 == 0:
                        print(f"已接收 {frame_count} 帧，当前FPS: {fps:.1f}")

                    # 检查退出键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        stop_receiving = True
                        break
                    elif key == ord('f'):  # 全屏切换
                        current_prop = cv2.getWindowProperty("Bamboo Detection", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("Bamboo Detection", cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN if current_prop == 0 else cv2.WINDOW_NORMAL)
                else:
                    print(f"解码帧失败，帧大小: {len(frame_data)} bytes")

            except ConnectionError as e:
                print(f"连接错误: {e}")
                break
            except Exception as e:
                print(f"处理帧时出错: {e}")
                # 清空缓冲区，避免错误数据影响后续帧
                buffer = b''

    except socket.timeout:
        print("连接服务器超时")
    except ConnectionRefusedError:
        print("连接被拒绝，服务器可能未启动")
    except Exception as e:
        print(f"接收视频时出错: {e}")
    finally:
        if client_socket:
            client_socket.close()
        cv2.destroyAllWindows()

        # 如果不是主动停止，尝试重新连接
        if not stop_receiving and reconnect_attempts < max_reconnect_attempts:
            reconnect_attempts += 1
            print(f"尝试重新连接 ({reconnect_attempts}/{max_reconnect_attempts})...")
            time.sleep(reconnect_delay)
            receive_and_display_video()
        else:
            print("程序结束")


def check_user_input():
    """检查用户输入，允许通过控制台命令退出"""
    global stop_receiving
    while not stop_receiving:
        try:
            command = input().strip().lower()
            if command == 'quit' or command == 'exit' or command == 'q':
                stop_receiving = True
                print("正在停止接收...")
                break
        except:
            break


if __name__ == "__main__":
    # 启动用户输入线程
    input_thread = threading.Thread(target=check_user_input, daemon=True)
    input_thread.start()

    # 开始接收视频
    receive_and_display_video()