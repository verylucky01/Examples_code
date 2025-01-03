def equalSubstring(s: str, t: str, maxCost: int) -> int:

    n, left, current_cost, max_length = len(s), 0, 0, 0

    for right in range(n):
        # 计算当前字符对的开销：
        current_cost += abs(ord(s[right]) - ord(t[right]))

        # 如果当前开销超过了预算，移动左指针：
        while current_cost > maxCost:
            current_cost -= abs(ord(s[left]) - ord(t[left]))
            left += 1

        # 更新最大长度：
        max_length = max(max_length, right - left + 1)

    return max_length, current_cost


# 测试用例：
s1, t1, maxCost1 = "abcd", "bcdf", 3
s2, t2, maxCost2 = "abcd", "cdef", 3
s3, t3, maxCost3 = "abcd", "wxyz", 2
s4, t4, maxCost4 = "abc", "abc", 1
s5, t5, maxCost5 = "a", "z", 25
print(equalSubstring(s1, t1, maxCost1))
print(equalSubstring(s2, t2, maxCost2))
print(equalSubstring(s3, t3, maxCost3))
print(equalSubstring(s4, t4, maxCost4))
print(equalSubstring(s5, t5, maxCost5))
