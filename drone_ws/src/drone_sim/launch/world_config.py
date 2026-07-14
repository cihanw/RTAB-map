# Single, shared world definition - sim.launch.py, slam.launch.py and
# bringup.launch.py import from here. Thus a world change is made from ONE
# place (see tasks/loop_closure_roadmap.md, Phase 1 integration note:
# "Defining a single world name variable managed from one place would be
# the cleanest approach").
#
# WORLD_NAME: The <world name="..."> value inside the .sdf file - Determines the prefix
# (/world/<WORLD_NAME>/...) of ALL scoped GZ->ROS topic paths.
# WORLD_PACKAGE / WORLD_FILE: In which package's share/ directory and with what name
# the .sdf file is located.
#
# To return to depot.sdf: WORLD_NAME='depot', WORLD_PACKAGE='drone_sim',
# WORLD_FILE='depot.sdf'.

WORLD_NAME = 'depot'
WORLD_PACKAGE = 'drone_sim'
WORLD_FILE = 'depot.sdf'
