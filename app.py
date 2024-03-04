import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk


class HandTrackingApp(ttk.Frame):
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracking App")

        # Create a custom style for a black-blue theme
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="black", foreground="blue")
        style.configure("TLabel", padding=6, background="#001F3F", foreground="white")

        
        # Create GUI components
        self.btn_open_camera = ttk.Button(root, text="Open Webcam", command=self.open_camera)
        self.btn_open_video = ttk.Button(root, text="Import Video", command=self.import_video)
        self.btn_quit = ttk.Button(root, text="Quit", command=self.quit_app)

        self.btn_open_camera.pack(pady=10)
        self.btn_open_video.pack(pady=10)
        self.btn_quit.pack(pady=10)

        # Initialize MediaPipe Hand module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Flags for controlling the app
        self.camera_opened = False
        self.video_path = ""

    def open_camera(self):
        if not self.camera_opened:
            self.camera_opened = True
            self.video_path = 0  # 0 corresponds to the default camera
            self.process_video()

    def import_video(self):
        self.video_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            self.process_video()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks using MediaPipe
            results = self.hands.process(rgb_frame)

            # Draw landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Display the frame
            cv2.imshow("Hand Tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()
        self.camera_opened = False

    def quit_app(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    
    # Simply set the theme
    root.tk.call("source", "Azure-ttk-theme-main/azure.tcl")
    root.tk.call("set_theme", "dark")
    
    app = HandTrackingApp(root)
    # app.pack(fill="both", expand=True)
    
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
    y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
    root.geometry("+{}+{}".format(x_cordinate, y_cordinate-20))
    root.mainloop()
