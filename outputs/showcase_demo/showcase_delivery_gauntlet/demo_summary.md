# Decision Showcase Summary

Selected planner: **clear_blocking_first**
Selected strategy: **clear_blocking_first_showcase_delivery_gauntlet**
Total score: **120**

## Candidate Strategies
- fixed_order: score=97, completed=5, failed=0, success=True
- nearest_first: score=107, completed=5, failed=0, success=True
- clear_blocking_first: score=120, completed=5, failed=0, success=True
- decision_pipeline: score=120, completed=5, failed=0, success=True

## Selected Action Sequence
- pick_place cube_center_anchor -> (16, 20)
- pick_place cube_near_left -> (15, 16)
- pick_place cube_far_blocked -> (15, 24)
- pick_place cube_far_right -> (15, 23)
- pick_place cube_near_right -> (17, 20)
