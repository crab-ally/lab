#!/usr/bin/env python3
import os
import mujoco
import numpy as np

MODEL_PATH=os.path.normpath(os.path.join(os.path.dirname(__file__),"../model/franka_emika_panda/panda.xml"))

def name(model,obj_type,idx):
    n=mujoco.mj_id2name(model,obj_type,int(idx))
    return n if n else f"<unnamed_{idx}>"

def joint_type(t):
    return {mujoco.mjtJoint.mjJNT_FREE:"FREE",mujoco.mjtJoint.mjJNT_BALL:"BALL",mujoco.mjtJoint.mjJNT_SLIDE:"SLIDE",mujoco.mjtJoint.mjJNT_HINGE:"HINGE"}.get(t,str(t))

def actuator_trntype(t):
    return {mujoco.mjtTrn.mjTRN_JOINT:"JOINT",mujoco.mjtTrn.mjTRN_JOINTINPARENT:"JOINTINPARENT",mujoco.mjtTrn.mjTRN_SLIDERCRANK:"SLIDERCRANK",mujoco.mjtTrn.mjTRN_TENDON:"TENDON",mujoco.mjtTrn.mjTRN_SITE:"SITE",mujoco.mjtTrn.mjTRN_BODY:"BODY",mujoco.mjtTrn.mjTRN_UNDEFINED:"UNDEFINED"}.get(t,str(t))

def print_section(title):
    print("\n"+"="*100)
    print(title)
    print("="*100)

def print_body_info(model,data):
    print_section("BODIES")
    total_mass=0.0
    for i in range(model.nbody):
        body_name=name(model,mujoco.mjtObj.mjOBJ_BODY,i)
        parent_id=model.body_parentid[i]
        parent_name=name(model,mujoco.mjtObj.mjOBJ_BODY,parent_id) if i!=0 else "WORLD"
        mass=float(model.body_mass[i])
        total_mass+=mass
        print(f"[{i:2d}] {body_name}")
        print(f"     parent      : {parent_name}")
        print(f"     local_pos   : {np.array2string(model.body_pos[i],precision=6,suppress_small=True)}")
        print(f"     local_quat  : {np.array2string(model.body_quat[i],precision=6,suppress_small=True)}")
        print(f"     world_pos   : {np.array2string(data.xpos[i],precision=6,suppress_small=True)}")
        print(f"     world_quat  : {np.array2string(data.xquat[i],precision=6,suppress_small=True)}")
        print(f"     mass        : {mass:.6f} kg")
    print(f"\nTOTAL MASS: {total_mass:.6f} kg")

def print_joint_info(model,data):
    print_section("JOINTS / QPOS / QVEL")
    for i in range(model.njnt):
        jname=name(model,mujoco.mjtObj.mjOBJ_JOINT,i)
        body_id=int(model.jnt_bodyid[i])
        body_name=name(model,mujoco.mjtObj.mjOBJ_BODY,body_id)
        qpos_addr=int(model.jnt_qposadr[i])
        qvel_addr=int(model.jnt_dofadr[i])
        jtype=joint_type(model.jnt_type[i])
        axis=model.jnt_axis[i]
        rng=model.jnt_range[i]
        limited=bool(model.jnt_limited[i])
        qpos_value=data.qpos[qpos_addr]
        qvel_value=data.qvel[qvel_addr]
        print(f"[{i:2d}] {jname}")
        print(f"     type        : {jtype}")
        print(f"     body        : {body_name} (id={body_id})")
        print(f"     qpos_addr   : {qpos_addr}")
        print(f"     qvel_addr   : {qvel_addr}")
        print(f"     qpos        : {qpos_value:.8f}")
        print(f"     qvel        : {qvel_value:.8f}")
        print(f"     axis        : {np.array2string(axis,precision=6,suppress_small=True)}")
        print(f"     limited     : {limited}")
        print(f"     range       : {np.array2string(rng,precision=6,suppress_small=True)}")

def print_arm_joints(model):
    print_section("PANDA ARM JOINTS")
    for joint_name in [f"joint{i}" for i in range(1,8)]:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
        if jid<0:
            print(f"{joint_name}: NOT FOUND")
            continue
        print(f"{joint_name}: joint_id={jid}, qpos_addr={model.jnt_qposadr[jid]}, qvel_addr={model.jnt_dofadr[jid]}, range={np.array2string(model.jnt_range[jid],precision=6,suppress_small=True)}")

def print_finger_joints(model):
    print_section("GRIPPER FINGER JOINTS")
    for joint_name in ["finger_joint1","finger_joint2"]:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
        if jid<0:
            print(f"{joint_name}: NOT FOUND")
            continue
        bid=int(model.jnt_bodyid[jid])
        print(f"{joint_name}: joint_id={jid}, body={name(model,mujoco.mjtObj.mjOBJ_BODY,bid)}, qpos_addr={model.jnt_qposadr[jid]}, qvel_addr={model.jnt_dofadr[jid]}, type={joint_type(model.jnt_type[jid])}, range={np.array2string(model.jnt_range[jid],precision=6,suppress_small=True)}")

def print_actuator_info(model):
    print_section("ACTUATORS")
    for i in range(model.nu):
        aname=name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i)
        trn_type=actuator_trntype(model.actuator_trntype[i])
        trn_id=int(model.actuator_trnid[i,0])
        target="-"
        if model.actuator_trntype[i] in [mujoco.mjtTrn.mjTRN_JOINT,mujoco.mjtTrn.mjTRN_JOINTINPARENT]:
            target=f"joint:{name(model,mujoco.mjtObj.mjOBJ_JOINT,trn_id)}"
        elif model.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_TENDON:
            target=f"tendon:{name(model,mujoco.mjtObj.mjOBJ_TENDON,trn_id)}"
        elif model.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_SITE:
            target=f"site:{name(model,mujoco.mjtObj.mjOBJ_SITE,trn_id)}"
        elif model.actuator_trntype[i]==mujoco.mjtTrn.mjTRN_BODY:
            target=f"body:{name(model,mujoco.mjtObj.mjOBJ_BODY,trn_id)}"
        print(f"[{i:2d}] {aname}")
        print(f"     transmission : {trn_type}")
        print(f"     target       : {target}")
        print(f"     ctrlrange    : {np.array2string(model.actuator_ctrlrange[i],precision=6,suppress_small=True)}")
        print(f"     forcerange   : {np.array2string(model.actuator_forcerange[i],precision=6,suppress_small=True)}")
        print(f"     gear         : {np.array2string(model.actuator_gear[i],precision=6,suppress_small=True)}")
        print(f"     gainprm      : {np.array2string(model.actuator_gainprm[i],precision=6,suppress_small=True)}")
        print(f"     biasprm      : {np.array2string(model.actuator_biasprm[i],precision=6,suppress_small=True)}")

def print_site_info(model,data):
    print_section("SITES / TCP CANDIDATES")
    if model.nsite==0:
        print("No sites defined in this Panda model.")
        print("TCP 후보는 hand body 또는 별도 site를 추가하여 사용하는 것이 좋습니다.")
        return
    for i in range(model.nsite):
        sname=name(model,mujoco.mjtObj.mjOBJ_SITE,i)
        body_id=int(model.site_bodyid[i])
        body_name=name(model,mujoco.mjtObj.mjOBJ_BODY,body_id)
        print(f"[{i:2d}] {sname}")
        print(f"     body        : {body_name} (id={body_id})")
        print(f"     local_pos   : {np.array2string(model.site_pos[i],precision=6,suppress_small=True)}")
        print(f"     world_pos   : {np.array2string(data.site_xpos[i],precision=6,suppress_small=True)}")
        print(f"     local_mat   : {np.array2string(model.site_mat0[i],precision=6,suppress_small=True)}")
        print(f"     world_mat   : {np.array2string(data.site_xmat[i].reshape(3,3),precision=6,suppress_small=True)}")

def tendon_type(t):
    return {mujoco.mjtTendon.mjTRN_FIXED:"FIXED",mujoco.mjtTendon.mjTRN_SPRING:"SPRING",mujoco.mjtTendon.mjTRN_SPATIAL:"SPATIAL"}.get(t,str(t))

def print_tendon_info(model):
    print_section("TENDONS")
    if model.ntendon==0:
        print("No tendons.")
        return
    for i in range(model.ntendon):
        tname=name(model,mujoco.mjtObj.mjOBJ_TENDON,i)
        print(f"[{i:2d}] {tname}")
        print(f"     length0     : {model.tendon_length0[i]:.8f}")
        print(f"     limited     : {bool(model.tendon_limited[i])}")
        print(f"     range       : {np.array2string(model.tendon_range[i],precision=6,suppress_small=True)}")
        print(f"     adr         : {model.tendon_adr[i]}")
        print(f"     num         : {model.tendon_num[i]}")

        adr=int(model.tendon_adr[i])
        num=int(model.tendon_num[i])

        print("     elements    :")
        for j in range(num):
            k=adr+j
            objtype=int(model.wrap_type[k])
            objid=int(model.wrap_objid[k])

            if objtype==int(mujoco.mjtObj.mjOBJ_JOINT):
                objname=name(model,mujoco.mjtObj.mjOBJ_JOINT,objid)
                print(f"       - joint: {objname}")
            elif objtype==int(mujoco.mjtObj.mjOBJ_GEOM):
                objname=name(model,mujoco.mjtObj.mjOBJ_GEOM,objid)
                print(f"       - geom: {objname}")
            elif objtype==int(mujoco.mjtObj.mjOBJ_SITE):
                objname=name(model,mujoco.mjtObj.mjOBJ_SITE,objid)
                print(f"       - site: {objname}")
            elif objtype==int(mujoco.mjtObj.mjOBJ_BODY):
                objname=name(model,mujoco.mjtObj.mjOBJ_BODY,objid)
                print(f"       - body: {objname}")
            else:
                print(f"       - object_type={objtype}, object_id={objid}")
                
def print_equality_info(model):
    print_section("EQUALITY CONSTRAINTS")
    if model.neq==0:
        print("No equality constraints.")
        return
    for i in range(model.neq):
        eq_type=model.eq_type[i]
        print(f"[{i:2d}] type={eq_type}")
        print(f"     obj1id={model.eq_obj1id[i]}")
        print(f"     obj2id={model.eq_obj2id[i]}")
        print(f"     active={bool(model.eq_active0[i])}")
        print(f"     data={np.array2string(model.eq_data[i],precision=6,suppress_small=True)}")

def print_gripper_info(model,data):
    print_section("GRIPPER")
    print("Panda gripper structure:")
    print("     hand")
    print("       ├── left_finger")
    print("       │     └── finger_joint1")
    print("       └── right_finger")
    print("             └── finger_joint2")
    print()
    for body_name in ["hand","left_finger","right_finger"]:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body_name)
        if bid>=0:
            print(f"{body_name}: body_id={bid}, world_pos={np.array2string(data.xpos[bid],precision=6,suppress_small=True)}")
    print()
    tendon_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TENDON,"split")
    actuator_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"actuator8")
    if tendon_id>=0:
        print(f"split tendon: tendon_id={tendon_id}")
    if actuator_id>=0:
        print(f"actuator8: actuator_id={actuator_id}, target=tendon:split, ctrlrange={np.array2string(model.actuator_ctrlrange[actuator_id],precision=6,suppress_small=True)}")
    print("finger_joint1 ↔ finger_joint2 equality constraint로 양쪽 finger가 동기화됩니다.")

def print_tcp_info(model,data):
    print_section("TCP / END-EFFECTOR")
    hand_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"hand")
    if hand_id>=0:
        print("TCP candidate body : hand")
        print(f"body_id            : {hand_id}")
        print(f"world position     : {np.array2string(data.xpos[hand_id],precision=6,suppress_small=True)}")
        print(f"world quaternion   : {np.array2string(data.xquat[hand_id],precision=6,suppress_small=True)}")
        print("※ 현재 공식 panda.xml에는 TCP 전용 site가 없습니다.")
        print("※ Pick & Place에서는 hand body 또는 hand에 TCP site를 추가하는 방식을 권장합니다.")
    else:
        print("hand body not found.")

def print_keyframes(model):
    print_section("KEYFRAMES")
    if model.nkey==0:
        print("No keyframes.")
        return
    for i in range(model.nkey):
        kname=name(model,mujoco.mjtObj.mjOBJ_KEY,i)
        print(f"[{i}] {kname}")
        print(f"     qpos={np.array2string(model.key_qpos[i],precision=6,suppress_small=True)}")
        print(f"     ctrl={np.array2string(model.key_ctrl[i],precision=6,suppress_small=True)}")

def print_qpos_qvel(model,data):
    print_section("CURRENT QPOS / QVEL")
    print(f"qpos ({model.nq}) = {np.array2string(data.qpos,precision=8,suppress_small=True)}")
    print(f"qvel ({model.nv}) = {np.array2string(data.qvel,precision=8,suppress_small=True)}")
    print()
    for i in range(model.njnt):
        jname=name(model,mujoco.mjtObj.mjOBJ_JOINT,i)
        qa=int(model.jnt_qposadr[i])
        va=int(model.jnt_dofadr[i])
        print(f"{jname}: qpos[{qa}]={data.qpos[qa]:.8f}, qvel[{va}]={data.qvel[va]:.8f}")

def main():
    print(f"[LOAD] {MODEL_PATH}")
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return
    try:
        model=mujoco.MjModel.from_xml_path(MODEL_PATH)
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print_section("MODEL INFO")
    print(f"MuJoCo version : {mujoco.__version__}")
    print("Model name    : Panda")
    print(f"nq            : {model.nq}")
    print(f"nv            : {model.nv}")
    print(f"nu            : {model.nu}")
    print(f"nbody         : {model.nbody}")
    print(f"njnt          : {model.njnt}")
    print(f"ngeom         : {model.ngeom}")
    print(f"nsite         : {model.nsite}")
    print(f"ncam          : {model.ncam}")
    print(f"ntendon       : {model.ntendon}")
    print(f"neq           : {model.neq}")
    print(f"nkey          : {model.nkey}")

    print_body_info(model,data)
    print_joint_info(model,data)
    print_arm_joints(model)
    print_finger_joints(model)
    print_actuator_info(model)
    print_site_info(model,data)
    print_tendon_info(model)
    print_equality_info(model)
    print_gripper_info(model,data)
    print_tcp_info(model,data)
    print_keyframes(model)
    print_qpos_qvel(model,data)

    print_section("DONE")
    print("Panda model information 출력 완료.")

if __name__=="__main__":
    main()