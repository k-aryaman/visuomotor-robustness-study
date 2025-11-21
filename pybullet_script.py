import pybullet as p
import pybullet_data
import numpy as np
import pickle
import time
import cv2

# ---------------- SETTINGS ----------------
SAVE_PATH = "demonstrations.pkl"
IMAGE_SIZE = (84, 84)
STEP_SIZE = 0.01       # EE movement per keypress
MAX_STEPS = 1000       # or run until ESC pressed
# ------------------------------------------

# ---------- HELPER FUNCTIONS ----------
def get_action_from_keys(keys):
    """Map keyboard keys to 4-dim action: [dx, dy, dz, gripper]"""
    dx = dy = dz = 0
    grip = 0

    if ord('w') in keys: dy = STEP_SIZE
    if ord('s') in keys: dy = -STEP_SIZE
    if ord('a') in keys: dx = -STEP_SIZE
    if ord('d') in keys: dx = STEP_SIZE
    if ord('r') in keys: dz = STEP_SIZE
    if ord('f') in keys: dz = -STEP_SIZE

    if ord('z') in keys: grip = 1       # open
    if ord('x') in keys: grip = -1      # close

    return np.array([dx, dy, dz, grip], dtype=np.float32)

def get_camera_image():
    width, height = IMAGE_SIZE
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.0, 0.0, 0.8],
        cameraTargetPosition=[0.5, 0, 0],
        cameraUpVector=[0, 0, 1],
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.1,
        farVal=2.0,
    )
    _, _, px, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER   # CPU-safe, works on Mac
    )
    rgb = np.reshape(px, (height, width, 4))[:, :, :3]
    return rgb

# ---------- SETUP PYBULLET ----------
physics = p.connect(p.DIRECT)   # headless, no GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
cube = p.loadURDF("cube_small.urdf", [0.6, 0, 0.05])

EE = 11  # end-effector link index
demos = []

print("Collecting demonstrations. Press ESC to quit.\n")

# ---------- MAIN LOOP ----------
step_count = 0
while step_count < MAX_STEPS:
    keys = p.getKeyboardEvents()
    if 27 in keys:   # ESC key
        break

    action = get_action_from_keys(keys)

    # Get current EE pose
    pos, orn = p.getLinkState(robot, EE)[0], p.getLinkState(robot, EE)[1]
    new_pos = np.array(pos) + action[:3]

    # Move via IK
    joint_angles = p.calculateInverseKinematics(robot, EE, new_pos, orn)
    for j in range(7):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joint_angles[j])

    # Gripper control
    if action[3] != 0:
        opening = 0.04 if action[3] > 0 else 0.0
        p.setJointMotorControl2(robot, 9, p.POSITION_CONTROL, opening)
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, opening)

    p.stepSimulation()
    time.sleep(1/240)

    # Capture camera image
    img = get_camera_image()

    # Optional: show in OpenCV window
    cv2.imshow("Simulation", img)
    cv2.waitKey(1)

    # Save to demonstrations list
    demos.append({"image": img, "action": action})
    step_count += 1

# ---------- SAVE DATA ----------
with open(SAVE_PATH, "wb") as f:
    pickle.dump(demos, f)

cv2.destroyAllWindows()
print(f"\nSaved {len(demos)} demonstrations to {SAVE_PATH}")