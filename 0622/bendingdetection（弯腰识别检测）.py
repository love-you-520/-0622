from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture("test.mp4")  #读取测试视频
fps = cap.get(cv2.CAP_PROP_FPS)  #得到视频帧率
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    "result.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)
frame_id = 0   #统计帧数来获取时间
log = [] #记录开始弯腰和结束弯腰的时间
history = {}  #表示人的状态，用来统计时间
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    t = frame_id / fps #算出当前的时间
    results = model(frame)
    if results[0].keypoints is not None:
        kpts_all = results[0].keypoints.xy.cpu().numpy()
        conf_all = results[0].keypoints.conf.cpu().numpy()
        for pid, kpts in enumerate(kpts_all):
            if pid not in history:
                history[pid] = {
                    "count": 0,  #弯腰持续的帧数
                    "bending": False,  #是否弯腰
                    "start": None  #开始时间
                }
            ls = kpts[5] #左肩
            rs = kpts[6] #右肩
            lh = kpts[11] #左髋关节
            rh = kpts[12] #右髋关节
            if np.min(conf_all[pid, [5, 6, 11, 12]]) < 0.5:
                continue
            shoulder = (ls + rs) / 2
            hip = (lh + rh) / 2
            dx = shoulder[0] - hip[0]
            dy = shoulder[1] - hip[1]
            angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
            if angle > 30:
                history[pid]["count"] += 1
            else:
                history[pid]["count"] = 0
            if history[pid]["count"] >= 5 and  history[pid]["bending"]==False:
                history[pid]["bending"] = True
                history[pid]["start"] = t
                log.append(f"Person{pid} START BENDING {t:.2f}s")
            if angle < 20 and history[pid]["bending"]:
                history[pid]["bending"] = False
                end = t
                start = history[pid]["start"]
                log.append(f"Person{pid} END BENDING {end:.2f}s")
            color = (0, 0, 255) if history[pid]["bending"] else (0, 255, 0)
            cv2.line(frame,
                     tuple(shoulder.astype(int)),
                     tuple(hip.astype(int)),
                     color, 3)
            cv2.putText(frame,
                        f"Person{pid} Angle:{angle:.1f}",
                        (int(shoulder[0]), int(shoulder[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)
            if history[pid]["bending"]:
                cv2.putText(frame,
                            f"BENDING，当前时间{t}",
                            (int(shoulder[0]), int(shoulder[1] - 50)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2)

        writer.write(frame)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    with open("log.txt", "w") as f:
        for l in log:
            f.write(l + "\n")
    print("已完成检测")