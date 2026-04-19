# Showcase Scene: showcase_delivery_gauntlet

A richer five-object tabletop delivery scene with mild blocking, asymmetric travel costs, and clearly suboptimal fixed ordering.

## Fixed Order
cube_far_blocked, cube_far_right, cube_center_anchor, cube_near_left, cube_near_right

## Objects
- cube_far_blocked: grid=(29, 24), goal=right_goal
- cube_far_right: grid=(28, 27), goal=right_goal
- cube_center_anchor: grid=(24, 21), goal=staging_goal
- cube_near_left: grid=(21, 18), goal=left_goal
- cube_near_right: grid=(22, 23), goal=staging_goal

## Goals
- left_goal: (15, 16)
- staging_goal: (16, 20)
- right_goal: (15, 24)

## Obstacles
- [(22, 20), (23, 20), (24, 20), (25, 21), (26, 22)]

## Forbidden Zones
- [(17, 21), (18, 21)]
