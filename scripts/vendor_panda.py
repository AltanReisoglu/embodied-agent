#!/usr/bin/env python
"""Vendor the Franka Panda from mujoco_menagerie, collision geometry only.

The hand-written arm this replaces was the source of three separate bugs: a capsule
intersecting the base so joint1 could never track its command, links too short to reach
the far half of the table, and a camera the arm stood in front of. None of those are
interesting problems, and none of them recur with a model DeepMind validated.

The reason it is vendored rather than referenced is that menagerie lives outside this
repo. The reason only the collision geometry is taken is size: the visual .obj meshes are
33MB, the collision meshes are 212KB, and the collision meshes *are* the link shapes --
simplified, but correct. The rendered arm looks blockier and the physics is identical.

    python scripts/vendor_panda.py --menagerie <path-to-mujoco_menagerie>

Source: https://github.com/google-deepmind/mujoco_menagerie (Apache 2.0)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

#: Between the fingertips, along the hand's approach axis. Everything in `skills/ik.py`
#: targets this site, so it is the one addition made to the upstream model.
GRASP_SITE_Z = 0.1034

#: Where the arm is bolted: the near edge of the table, surface at z=0.40. Objects sit
#: 0.45-0.65m out, comfortably inside the Panda's 0.855m reach.
MOUNT = "0 -0.32 0.40"


def vendor(menagerie: Path, out: Path, mount: str = MOUNT) -> None:
    src = menagerie / "franka_emika_panda"
    if not (src / "panda.xml").exists():
        raise SystemExit(f"no panda.xml under {src}")

    tree = ET.parse(src / "panda.xml")
    root = tree.getroot()

    # 1. Drop every visual geom. They carry the 33MB of .obj meshes and contribute
    #    nothing the physics or the depth buffer depends on.
    dropped = 0
    for parent in root.iter():
        for geom in [g for g in parent.findall("geom") if g.get("class") == "visual"]:
            parent.remove(geom)
            dropped += 1

    # 2. Keep only the mesh assets something still points at.
    referenced = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    kept_files: list[str] = []
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name") or Path(mesh.get("file", "")).stem
        if name in referenced:
            kept_files.append(mesh.get("file"))
        else:
            asset.remove(mesh)

    # 3. Add the grasp site. Upstream has no site at the tool centre point, and the IK
    #    solver needs one to drive.
    hand = next((b for b in root.iter("body") if b.get("name") == "hand"), None)
    if hand is None:
        raise SystemExit("could not find the hand body")
    ET.SubElement(
        hand,
        "site",
        {"name": "grasp", "pos": f"0 0 {GRASP_SITE_Z}", "size": "0.008", "rgba": "1 0 0 0.4"},
    )

    # 4. Drop the upstream keyframe. A scene that adds objects has free joints of its
    #    own, so a 9-element qpos no longer matches and MuJoCo refuses to compile.
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)

    # 5. Mount it. `<include>` splices bodies straight into the scene's worldbody, so
    #    there is nowhere else to say where the robot stands; link0 has no `pos` of its
    #    own upstream and would otherwise sit at the origin, under the table.
    worldbody = root.find("worldbody")
    link0 = next((b for b in worldbody.findall("body") if b.get("name") == "link0"), None)
    if link0 is None:
        raise SystemExit("could not find link0")
    link0.set("pos", mount)
    # The scene lights the table itself; this one would double-expose it.
    for light in worldbody.findall("light"):
        worldbody.remove(light)

    # 6. Grip. Upstream leaves the fingertip pads on MuJoCo's default friction (1.0) with
    #    no damping on the finger joints, which is fine for reaching demos and not fine
    #    for carrying: the cube slipped out of the fingers on every transfer. A real
    #    Franka has rubber pads, so raising this is closer to the hardware, not further.
    for default in root.iter("default"):
        cls = default.get("class") or ""
        if cls.startswith("fingertip_pad_collision"):
            for geom in default.findall("geom"):
                geom.set("friction", "2.5 0.2 0.01")
                geom.set("solimp", "0.97 0.995 0.001")
        elif cls == "finger":
            for joint in default.findall("joint"):
                joint.set("damping", "20")

    # 7. Drop the <compiler> element. Under `<include>` MuJoCo resolves meshdir against
    #    the *including* file, so upstream's `meshdir="assets"` fights whatever the scene
    #    declares. The scene owns compiler settings; this file is include-only.
    for compiler in root.findall("compiler"):
        root.remove(compiler)

    # 8. Write it out next to the meshes it kept.
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    for filename in kept_files:
        shutil.copy2(src / "assets" / filename, out / "assets" / filename)
    shutil.copy2(src / "LICENSE", out / "LICENSE")

    ET.indent(tree, space="  ")
    tree.write(out / "panda.xml", encoding="utf-8", xml_declaration=False)

    size = sum(f.stat().st_size for f in (out / "assets").iterdir())
    print(f"dropped {dropped} visual geoms")
    print(f"kept {len(kept_files)} collision meshes ({size / 1024:.0f} KB)")
    print(f"wrote {out / 'panda.xml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--menagerie",
        default="yardimci-repolar/labs/mujoco_menagerie",
        help="path to a mujoco_menagerie checkout",
    )
    parser.add_argument("--out", default="src/embodied_agent/assets/panda")
    parser.add_argument("--mount", default=MOUNT, help="where link0 is bolted")
    args = parser.parse_args()
    vendor(Path(args.menagerie).resolve(), Path(args.out).resolve(), args.mount)
    return 0


if __name__ == "__main__":
    sys.exit(main())
