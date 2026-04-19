# **CHANGELOG**
## *v1.1.0*  
**Date:** 2 April 2026  
**Description:**  
FEATURE UPDATE: Added r2livecode.py - Navigation Algorithm for Final Maze

## *v1.2.0*  
**Date:** 7 April 2026  
**Description:**  
FEATURE UPDATE: Added r2dockingtest.py for robot docking [WORK IN PROGRESS]  
REFINEMENTS: Parameter Updates to r2livecode for better navigation reliability

## *v1.3.0*  
**Date:** 11 April 2026  
**Description:**  
FEATURE UPDATE: Merged r2dockingtest functionality into r2livecode to enable nav & dock sequence [NEED TO REFINE]

## *v1.4.0*  
**Date:** 13 April 2026  
**Description:**  
FEATURE UPDATE: Added fulltest(polardocking).py for refined docking algorithm and updated docking and navigation parameters;  
Changed ArUco Marker Station Detection Logic - Assigned ID Numbers (0: A; 1: B)

## *v1.5.0*  
**Date:** 14 April 2026  
**Description:**  
FEATURE UPDATE: Added newtest2.py for quick and easy testing of proposed docking & navigation algorithms  
REFINEMENTS: Fixed a typo in CHANGELOG.md; Fine-tuned parameters for docking and navigation

## *v1.5.1*  
**Date:** 14 April 2026  
**Description:**  
REFINEMENTS: Updated Navigation Algorithm & Parameters to be more robust. Simplified Docking Logic. Renamed newtest.py to r2CDE2310_FINAL.py.

## *v1.5.2*  
**Date:** 14 April 2026  
**Description:**  
REFINEMENTS: Removed checks to see if both stations are completed to allow the bot to continue to navigate in the event of not full map completion. Removed 'MISSION_COMPLETE' state.  
**NOTE! THIS IS THE CODE USED IN EVALUATION**

## *v1.5.3*  
**Date:** 14 April 2026  
**Description:**  
REFINEMENTS: On the fly parameter change during final run for r2CDE2310_FINAL.py Navigation

## *v1.5.4*
**Date:** 14 April 2026  
**Description:**  
FEATURE UPDATE: Added test files for algorithm research and comparison — r2livecode_v2.py (wall inflation + scored frontier + wall-safe smoothing), fulltest(astar).py (A* navigation), fulltest(astar+polardocking).py (A* + polar arc docking), fulltest(astar+3phasedocking).py (A* + 3-phase LiDAR docking), fulltest(3phasedocking).py (3-phase docking standalone), fulltest(polardocking_standalone).py (polar docking standalone)
