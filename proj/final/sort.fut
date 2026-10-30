import "lib/github.com/diku-dk/sorts/radix_sort"

def rs_sort [n] (xs: [n]u32) : [n]u32 =
  radix_sort u32.num_bits u32.get_bit xs

-- timed entry point
entry rs [n] (xs: [n]u32) : [n]u32 = rs_sort xs

-- correctness check 
def is_sorted [n] (xs: [n]u32) : bool =
  let m = i64.max 0 (n - 1)
  in and (map (\i -> xs[i] <= xs[i + 1]) (iota m))

def same_content [n] (xs: [n]u32) (ys: [n]u32) : bool =
  (reduce (+) 0u64 (map u64.u32 xs) == reduce (+) 0u64 (map u64.u32 ys))
  && (reduce (^) 0u32 xs == reduce (^) 0u32 ys)

entry rs_ok [n] (xs: [n]u32) : bool =
  let ys = rs_sort xs in is_sorted ys && same_content xs ys
