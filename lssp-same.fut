-- Parallel Longest Satisfying Segment
--
-- ==
-- entry: main seq
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
--
-- compiled input {
--    [5i32,7i32,4i32,3i32,2i32,3i32,4i32,6i32,7i32,-3i32,4i32,6i32,8i32,6i32,6i32,6i32]
-- }  
-- output { 
--    3
-- }
--
-- compiled random input { [1000000]i32 }
--
-- compiled random input { [5000000]i32 }

import "lssp"
import "lssp-seq"

let same_pred1 _   = true
let same_pred2 (x: i32) (y: i32) = (x == y)

entry main (xs: []i32) : i32 = lssp     same_pred1 same_pred2 xs
entry seq  (xs: []i32) : i32 = lssp_seq same_pred1 same_pred2 xs
