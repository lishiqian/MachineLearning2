# # Definition for singly-linked list.
# # class ListNode(object):
# #     def __init__(self, x):
# #         self.val = x
# #         self.next = None
# #
# #     def build(list):
# #         temp_head = ListNode(0)
# #         cur = temp_head
# #         for it in list:
# #             cur.next = ListNode(it)
# #             cur = cur.next
# #
# #         return temp_head.next
# #
# # class Solution(object):
# #     def reverseKGroup(self, head, k):
# #         """
# #         :type head: ListNode
# #         :type k: int
# #         :rtype: ListNode
# #         """
# #         temp_head = ListNode(0)
# #         temp_head.next = head
# #
# #         prv = temp_head
# #         fast_prv = temp_head
# #         while fast_prv:
# #             for i in range(k):
# #                 if fast_prv:
# #                     fast_prv = fast_prv.next
# #                 else:
# #                     return temp_head.next
# #
# #             if fast_prv:
# #                 self.reverse_list(prv, fast_prv.next)
# #                 prv = fast_prv
# #
# #         return temp_head.next
# #
# #     def reverse_list(self, head, end_node):
# #         cur = head.next
# #         head.next = end_node
# #         while cur != end_node:
# #             cur_next = cur.next
# #
# #             cur.next = head.next
# #             head.next = cur
# #
# #             cur = cur_next
# #
# #
# # head = ListNode.build([1,2,3,4,5])
# # s = Solution()
# # s.reverseKGroup(head,2)

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(needle) == 0:
            return 0

        h_len = len(haystack)
        n_len = len(needle)
        for i in range(h_len):
            l = 0
            for j in range(n_len):
                if (l == n_len - 1):
                    return i
                if i + l < h_len and haystack[i + l] == haystack[j]:
                    l += 1
                else:
                    break

        return -1

s = Solution()
index = s.strStr('hello','ll')
print(index)