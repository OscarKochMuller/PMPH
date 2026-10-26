import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/sorts/merge_sort"

def rs_sort [n] (xs: [n]u32) : [n]u32 =
  radix_sort u32.num_bits u32.get_bit xs

def ms_sort [n] (xs: [n]u32) : [n]u32 =
  merge_sort (u32.<=) xs

-- timed entry points
entry rs [n] (xs: [n]u32) : [n]u32 = rs_sort xs
entry ms [n] (xs: [n]u32) : [n]u32 = ms_sort xs

-- correctness checks (not timed): output is sorted and has the same sum and xor as the input
def is_sorted [n] (xs: [n]u32) : bool =
  let m = i64.max 0 (n - 1)
  in and (map (\i -> xs[i] <= xs[i + 1]) (iota m))

def same_content [n] (xs: [n]u32) (ys: [n]u32) : bool =
  (reduce (+) 0u64 (map u64.u32 xs) == reduce (+) 0u64 (map u64.u32 ys))
  && (reduce (^) 0u32 xs == reduce (^) 0u32 ys)

entry rs_ok [n] (xs: [n]u32) : bool =
  let ys = rs_sort xs in is_sorted ys && same_content xs ys

entry ms_ok [n] (xs: [n]u32) : bool =
  let ys = ms_sort xs in is_sorted ys && same_content xs ys
