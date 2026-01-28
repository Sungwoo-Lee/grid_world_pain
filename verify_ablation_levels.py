
import os
import sys
import yaml
import glob
from src.environment import GridWorld
from src.environment.sensor import SensorySystem
from src.utils.config import get_default_config

def verify_ablation_levels():
    # Define expected configuration for each level based on the audit
    expected_configs = {
        '01_goal_only.yaml': {
            'max_steps': 200, 'relocate': False, 'health': False, 'satiation': False, 'homeo': False,
            'sensory': False, 'pain': False, 'olfac': False, 'proprio': False, 'collision': False, 'loc': False
        },
        '02_danger_only.yaml': {
            'max_steps': 200, 'relocate': False, 'health': True, 'satiation': False, 'homeo': False,
            'sensory': False, 'pain': False, 'olfac': False, 'proprio': False, 'collision': False, 'loc': False
        },
        '03_dynamic_fixed.yaml': {
            'max_steps': 200, 'relocate': False, 'health': True, 'satiation': False, 'homeo': False,
            'sensory': False, 'pain': False, 'olfac': False, 'proprio': False, 'collision': False, 'loc': False
        },
        '04_nociception.yaml': {
            'max_steps': 200, 'relocate': False, 'health': True, 'satiation': False, 'homeo': False,
            'sensory': True, 'pain': True, 'olfac': False, 'proprio': False, 'collision': False, 'loc': False
        },
        '05_olfactory.yaml': {
            'max_steps': 500, 'relocate': True, 'health': True, 'satiation': True, 'homeo': False,
            'sensory': True, 'pain': True, 'olfac': True, 'proprio': False, 'collision': False, 'loc': False
        },
        '06_proprioception.yaml': {
            'max_steps': 500, 'relocate': True, 'health': True, 'satiation': True, 'homeo': False,
            'sensory': True, 'pain': True, 'olfac': True, 'proprio': True, 'collision': False, 'loc': False
        },
        '07_collision.yaml': {
            'max_steps': 500, 'relocate': True, 'health': True, 'satiation': True, 'homeo': False,
            'sensory': True, 'pain': True, 'olfac': True, 'proprio': True, 'collision': True, 'loc': False
        },
        '08_homeostatic.yaml': {
            'max_steps': 500, 'relocate': True, 'health': True, 'satiation': True, 'homeo': True,
            'sensory': True, 'pain': True, 'olfac': True, 'proprio': True, 'collision': True, 'loc': False
        },
        '09_location.yaml': {
            'max_steps': 500, 'relocate': True, 'health': True, 'satiation': True, 'homeo': True,
            'sensory': True, 'pain': True, 'olfac': True, 'proprio': True, 'collision': True, 'loc': True
        },
    }

    print("=== Ablation Study Level Verification ===\n")
    
    config_files = sorted(glob.glob("configs/ablation/*.yaml"))
    
    for config_path in config_files:
        filename = os.path.basename(config_path)
        if filename not in expected_configs:
            print(f"[SKIP] {filename} (Not in expected list)")
            continue
            
        expected = expected_configs[filename]
        
        # 1. Load Base Config
        config = get_default_config()
        
        # 2. Simulate Train.py Merging Logic
        # (Train logic: overrides defaults)
        with open(config_path, 'r') as f:
            override_config = yaml.safe_load(f)
            config.merge(override_config)
            
        # 3. Extract Resolved Parameters
        max_steps = config.get_mandatory('environment.max_steps')
        relocate = config.get_mandatory('environment.relocate_resource')
        
        health = config.get_mandatory('body.with_health')
        satiation = config.get_mandatory('body.with_satiation')
        homeo = config.get_mandatory('body.use_homeostatic_reward')
        
        using_sensory = config.get_mandatory('sensory.using_sensory')
        pain = config.get('sensory.nociception_enabled', True)
        olfac = config.get('sensory.olfactory_enabled', True)
        proprio = config.get('sensory.proprioception_enabled', False)
        collision = config.get('sensory.collision_sensor_enabled', True)
        location = config.get('sensory.location_sensor', True) # Defaults to true in base
        
        # 4. Compare
        errors = []
        
        if max_steps != expected['max_steps']: errors.append(f"MaxSteps: {max_steps} != {expected['max_steps']}")
        if relocate != expected['relocate']: errors.append(f"Relocate: {relocate} != {expected['relocate']}")
        if health != expected['health']: errors.append(f"Health: {health} != {expected['health']}")
        if satiation != expected['satiation']: errors.append(f"Satiation: {satiation} != {expected['satiation']}")
        if homeo != expected['homeo']: errors.append(f"HomeoRew: {homeo} != {expected['homeo']}")
        if using_sensory != expected['sensory']: errors.append(f"UsingSensory: {using_sensory} != {expected['sensory']}")
        
        # Only check sensor toggles if sensory system is active
        if using_sensory:
            if pain != expected['pain']: errors.append(f"Pain: {pain} != {expected['pain']}")
            if olfac != expected['olfac']: errors.append(f"Olfactory: {olfac} != {expected['olfac']}")
            if proprio != expected['proprio']: errors.append(f"Proprio: {proprio} != {expected['proprio']}")
            if collision != expected['collision']: errors.append(f"Collision: {collision} != {expected['collision']}")
            if location != expected['loc']: errors.append(f"Location: {location} != {expected['loc']}")
            
        # 5. Report
        if not errors:
            print(f"✅ {filename}: PASSED")
        else:
            print(f"❌ {filename}: FAILED")
            for e in errors:
                print(f"   - {e}")

if __name__ == "__main__":
    verify_ablation_levels()
