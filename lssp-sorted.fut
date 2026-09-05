-- Parallel Longest Satisfying Segment
--
-- ==
-- entry: main seq
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
--
-- compiled input {
--    [5,7,4,3,2,3,4,6,7,-3,4,6,8,6,6,6]
-- }  
-- output { 
--    5
-- }

import "lssp"
import "lssp-seq"

let sorted_pred1 _   = true
let sorted_pred2 (x: i32) (y: i32) = (x <= y)

entry main (xs: []i32) : i32 = lssp     sorted_pred1 sorted_pred2 xs
entry seq  (xs: []i32) : i32 = lssp_seq sorted_pred1 sorted_pred2 xs
