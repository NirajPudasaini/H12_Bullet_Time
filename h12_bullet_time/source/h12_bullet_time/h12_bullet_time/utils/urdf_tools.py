import xml.etree.ElementTree as ET
from collections import defaultdict


class Pose3D:
    def __init__(self, pos: tuple[float, float, float], quat: tuple[float, float, float, float]):
        self.pos = pos
        self.quat = quat


def extract_sensor_positions_from_urdf(urdf_path: str, debug: bool = False) -> dict[str, list[tuple[float, float, float]]]:
    """Extract sensor positions from a URDF file.
    
    Parses the URDF to find all sensor joints (joints with '_sensor_' in their name),
    extracts their positions, and groups them by the rigid body link they're attached to.
    For skin links, this function finds the parent rigid body link.
    
    Args:
        urdf_path: Path to the URDF file
        debug: If True, print debug information
        
    Returns:
        Dictionary mapping rigid body link names to lists of sensor positions (x, y, z tuples)
        Example: {"torso_link": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)], ...}
    """
    if debug:
        print(f"[DEBUG URDF]: Parsing URDF file: {urdf_path}")
    
    # Parse the URDF XML file
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # First, build a map of child link -> parent link relationships
    link_parent_map = {}
    for joint in root.findall('joint'):
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        if parent_elem is not None and child_elem is not None:
            parent_link = parent_elem.get('link', '')
            child_link = child_elem.get('link', '')
            link_parent_map[child_link] = parent_link
    
    if debug:
        print(f"[DEBUG URDF]: Built parent map with {len(link_parent_map)} entries")
    
    # Dictionary to store sensors grouped by parent link
    sensors_by_link = defaultdict(list)
    sensor_count = 0
    
    # Find all joints with '_sensor_' in their name
    for joint in root.findall('joint'):
        joint_name = joint.get('name', '')
        
        # Check if this is a sensor joint
        if '_sensor_' in joint_name:
            sensor_count += 1
            # Get the parent link of the sensor
            parent_elem = joint.find('parent')
            if parent_elem is not None:
                parent_link = parent_elem.get('link', '')
                
                # If the parent link is a skin link, find its parent (the actual rigid body)
                if parent_link in link_parent_map:
                    rigid_body_link = link_parent_map[parent_link]
                    if debug and sensor_count <= 5:  # Only print first few
                        print(f"[DEBUG URDF]: Sensor {joint_name}: {parent_link} -> {rigid_body_link}")
                else:
                    rigid_body_link = parent_link
                    if debug and sensor_count <= 5:
                        print(f"[DEBUG URDF]: Sensor {joint_name}: {parent_link} (no parent)")
                
                # Get the origin xyz position
                origin_elem = joint.find('origin')
                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    # Parse the xyz string "x y z" into a tuple of floats
                    xyz = tuple(map(float, xyz_str.split()))
                    
                    # Add this sensor position to the rigid body link's list
                    sensors_by_link[rigid_body_link].append(xyz)
    
    if debug:
        print(f"[DEBUG URDF]: Found {sensor_count} total sensors across {len(sensors_by_link)} links")
    
    # Convert defaultdict to regular dict
    return dict(sensors_by_link)
    