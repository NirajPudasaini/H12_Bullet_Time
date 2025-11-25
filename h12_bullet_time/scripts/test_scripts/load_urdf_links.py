#!/usr/bin/env python3
"""Simple script to load H1-2 URDF and print all links."""

import xml.etree.ElementTree as ET

# Path to the URDF file
urdf_path = "/home/niraj/isaac_projects/H12_Bullet_Time/h12_bullet_time/source/h12_bullet_time/h12_bullet_time/assets/robots/gentact_descriptions/robots/h1-2/h1_2_torso_skin.urdf"

try:
    # Parse the URDF XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    print("=" * 80)
    print(f"URDF File: {urdf_path}")
    print("=" * 80)
    
    # Get all links
    links = root.findall('link')
    print(f"\nTotal Links: {len(links)}\n")
    
    print("Link Names:")
    print("-" * 80)
    for i, link in enumerate(links, 1):
        link_name = link.get('name')
        print(f"{i:3d}. {link_name}")
    
    # Get all joints
    joints = root.findall('joint')
    print(f"\n\nTotal Joints: {len(joints)}\n")
    
    print("Joint Names and Types:")
    print("-" * 80)
    for i, joint in enumerate(joints, 1):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        print(f"{i:3d}. {joint_name:40s} ({joint_type:10s}) {parent} → {child}")
    
    print("\n" + "=" * 80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
