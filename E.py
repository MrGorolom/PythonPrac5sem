def find_path_sums(tree):
    if tree == None:
        return
    stack = list()
    stack.append(0)
    stack.append(tree)
    dinamic_sum = list()
    curr_sum = 0
    curr_node = tree
    high = 0
    flag = True
    non_exit_flag = True
    while len(stack) > 0 or non_exit_flag:
        non_exit_flag = True
        if flag:
            curr_node = stack.pop(-1)
            high = stack.pop(-1)
            if len(dinamic_sum) > high:
                curr_sum = dinamic_sum[high]
        curr_sum += curr_node[0]
        if len(dinamic_sum) > high:
            dinamic_sum[high] = curr_sum
        else:
            dinamic_sum.append(curr_sum)
        if curr_node[1] == None:
            if curr_node[2] == None:
                non_exit_flag = False
                print(curr_sum)
                #dinamic_sum[high] -= curr_node[0]
                flag = True
            else:
                curr_node = curr_node[2]
                high += 1
                flag = False
        else:
            if curr_node[2] != None:
                stack.append(high)
                stack.append(curr_node[2])
            curr_node = curr_node[1]
            high += 1
            flag = False
