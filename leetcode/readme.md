# 20260114

## 两数之和

```
# 暴力
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
         n = len(nums)
         for i in range(n):
             for j in range(i+1,n):
                 if nums[i] + nums[j] == target:
                     return [i,j]
        

# 哈希
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:        
        hashtable = dict()
        for i,num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num],i]
            hashtable[nums[i]] = i
        return [] 
```

##  字母异位词

```
# 初始方法
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        final_answer = []
        used = set()

        for word in strs:
            if word in used:
                continue
            answer = []
            for i in strs:
                if sorted(word) == sorted(i):
                    answer.append(i)
                    used.add(i)
            final_answer.append(answer)
                
        return final_answer
        
        
# 使用排序的方法 使用哈希思想
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = defaultdict(list)
        for s in strs:
            key = ''.join(sorted(s))
            mp[key].append(s)
        return list(mp.values())


# 最长序列
```



https://blog.csdn.net/wjj2586590669/article/details/126392080

分类讨论

不同数字

相同数字





















































































