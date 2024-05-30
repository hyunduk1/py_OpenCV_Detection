import cv2
import matplotlib.pyplot as plt

# 카메라
cap = cv2.VideoCapture(0)

# 차트 초기화
plt.figure(figsize=(10, 6))
plt.title('Pixel Values at y=150')
plt.xlabel('x')
plt.ylabel('Pixel Value')
plt.ion()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_zoomed = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    row_values = frame_zoomed[300, :, 0]
    threshold_value = 100

    plt.clf()
    plt.plot(row_values, color='b', lw=2)

    # 스레시홀드 값 선 표시
    plt.axhline(y=threshold_value, color='r', linestyle='--', label='Threshold Value')

    plt.pause(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.ioff()
plt.close()

cap.release()
cv2.destroyAllWindows()