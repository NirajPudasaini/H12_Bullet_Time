"""
TOF Sensor Readings Script
Loads the H12 URDF and extracts TOF (Time-of-Flight) sensor link information.

Note: Sensor links are defined in the URDF but get merged during USD conversion
because they have no mass/inertia. We extract their transforms directly from URDF.
"""

import torch
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

# URDF path
urdf_path = Path(__file__).parent.parent.parent / "source" / "h12_bullet_time" / "h12_bullet_time" / "assets" / "robots" / "gentact_descriptions" / "robots" / "h1-2" / "h1_2_torso_skin.urdf"

print(f"Loading URDF from: {urdf_path}")
print(f"URDF exists: {urdf_path.exists()}")

# Parse URDF directly to extract sensor link definitions
tree = ET.parse(str(urdf_path))
root = tree.getroot()

# Extract all links
all_links = {}
for link in root.findall('.//link'):
    link_name = link.get('name')
    all_links[link_name] = link

# Extract all joints and their transforms
joint_transforms = {}
for joint in root.findall('.//joint'):
    joint_name = joint.get('name')
    joint_type = joint.get('type')
    child_link = joint.find('child').get('link')
    parent_link = joint.find('parent').get('link')
    origin = joint.find('origin')
    
    if origin is not None:
        pos = origin.get('xyz', '0 0 0').split()
        pos = np.array([float(x) for x in pos])
        rpy = origin.get('rpy', '0 0 0').split()
        rpy = np.array([float(x) for x in rpy])
        
        joint_transforms[child_link] = {
            'parent': parent_link,
            'position': pos,
            'rpy': rpy,
            'joint_type': joint_type
        }

# Now launch Isaac Sim for visualization
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
app = app_launcher.app

# Import after Isaac Sim is launched
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS

# Setup simulation
sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=1)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])

# Create and spawn the robot in simulation
robot_cfg = H12_CFG_HANDLESS.replace(prim_path="/World/Robot")
robot = Articulation(cfg=robot_cfg)

# Reset simulation to initialize the robot
sim.reset()

# Simulation loop
print("\n" + "="*80)
print("TOF SENSOR LINKS IN H12 ROBOT (from URDF)")
print("="*80)

# Filter sensor links from URDF
sensor_links_urdf = {name: link for name, link in all_links.items() if 'sensor' in name.lower()}
print(f"\nTotal sensor links in URDF: {len(sensor_links_urdf)}\n")

# Group sensors by parent
sensor_groups = {}
for sensor_name in sensor_links_urdf.keys():
    if sensor_name in joint_transforms:
        parent = joint_transforms[sensor_name]['parent']
    else:
        parent = "unknown"
    
    if parent not in sensor_groups:
        sensor_groups[parent] = []
    sensor_groups[parent].append(sensor_name)

# Print organized sensor structure from URDF
print("Sensor Links (organized by parent in URDF):\n")
for parent_name in sorted(sensor_groups.keys()):
    sensors = sensor_groups[parent_name]
    print(f"  {parent_name}: ({len(sensors)} sensors)")
    for sensor in sorted(sensors)[:5]:
        print(f"    - {sensor}")
    if len(sensors) > 5:
        print(f"    ... and {len(sensors) - 5} more")

print("\n" + "="*80)
print("SENSOR POSITIONS FROM URDF TRANSFORMS")
print("="*80 + "\n")

# Simulate and visualize
for step in range(20):
    sim.step()
    
    if step == 10:  # Print positions after a few steps
        print("Sensor Link Transform Data (from URDF):\n")
        
        print(f"{'Sensor Name':<40} {'Parent Link':<30} {'Position (XYZ)':>25}")
        print("-" * 100)
        
        # Print by parent group
        for parent_name in sorted(sensor_groups.keys()):
            sensors = sorted(sensor_groups[parent_name])
            print(f"\n{parent_name}:")
            
            for sensor_name in sensors[:10]:
                try:
                    transform = joint_transforms.get(sensor_name, {})
                    parent = transform.get('parent', 'unknown')
                    pos = transform.get('position', np.array([0, 0, 0]))
                    
                    print(f"  {sensor_name:<38} {parent:<28} ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")
                except Exception as e:
                    print(f"  {sensor_name:<38} ERROR - {e}")
            
            if len(sensors) > 10:
                print(f"  ... and {len(sensors) - 10} more sensors in {parent_name}")

print("\n" + "="*80)
print("SENSOR DATA IN SIMULATION (Actual Body Count)")
print("="*80)

print("\n" + "="*80)
print("SENSOR DATA IN SIMULATION (Actual Body Count)")
print("="*80)

# Get simulated body names (after merging)
body_names = robot.body_names
print(f"\nTotal bodies in simulation: {len(body_names)}")
print("(Individual sensor links merged into parent bodies during USD conversion)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
URDF Definition:
- Total sensor links defined: {len(sensor_links_urdf)}
- Sensor groups: {len(sensor_groups)}

Simulation State:
- Total bodies: {len(body_names)}
- Sensor links in simulation: 0 (merged into parents)

Note:
Sensor links are defined in the URDF but don't appear as separate bodies in 
the simulation because they have no mass/inertia/collisions. During USD conversion,
they are merged into their parent links. However, their original transform data
is preserved in the URDF and can be used to calculate sensor positions.

For TOF sensor functionality:
1. Use the URDF joint transforms to get sensor frame locations
2. Calculate absolute positions by forward kinematics from parent links
3. Use distance to robot links as approximate depth readings
4. Or implement raycasting from sensor frames toward obstacles
""")

print("\nDone!")
