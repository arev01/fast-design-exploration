class Solution:
    def searchRange(self, nums, target):
        left, right = self.bisect_left(nums, target), self.bisect_right(nums, target) - 1
        return [
            left if left < len(nums) and nums[left] == target else -1,
            right if 0 <= right < len(nums) and nums[right] == target else -1
        ]
      
    def bisect_left(self, a, x):
        """returns the largest i where all a[:i] is less than x"""
        lo, hi = 0, len(a)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if a[mid] < x: lo = mid + 1
            else: hi = mid
        return lo
    
    def bisect_right(self, a, x):
        """returns the largest i where all a[:i] is less than or equal to x"""
        lo, hi = 0, len(a)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if a[mid] <= x: lo = mid + 1
            else: hi = mid
        return lo
    
def take_closest(myList, myNumber):
    """returns closest value to myNumber"""

    pos = Solution().bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before