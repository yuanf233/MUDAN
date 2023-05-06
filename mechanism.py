from cmath import nan
from dis import dis
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from copy import deepcopy
import copy

inf = 10000
nodes = 4039

f = pd.read_csv('facebook_combined.txt', index_col=False, header=None, sep=' ')
f.columns = ['begin', 'end']
df1 = f.groupby('begin').end.apply(list).to_dict()
df2 = f.groupby('end').begin.apply(list).to_dict()
for i in df2.keys():
    if df1.get(i) != None:
        df1[i] = df1[i] + df2[i]
    else:
        df1[i] = df2[i]
relation = df1

for i in range(4039):
    if (df1.get(i) == None):
        print(i)


def generate(initial, supply, model):
    v = {}
    m = supply
    match model:
        case 1:
            a = np.random.randint(low=1, high=200001, size=nodes)
        case 2:
            a = []
        case 3:
            a = pd.read_csv('valuation.csv', index_col=0)
            a = a[-1:]
            a = np.array(a).tolist()[0]
    if supply <= 10:
        b = np.random.randint(low=1, high=supply + 1, size=nodes)
    else:
        b = np.random.randint(low=1, high=11, size=nodes)
    df = relation
    k = 0
    for i in df.keys():
        match model:
            case 1:
                tmp = np.random.randint(low=0, high=a[k], size=b[k])
            case 2:
                tmp = np.random.randint(low=0, high=200001, size=b[k])
            case 3:
                tmp = np.random.randint(low=0, high=a[k], size=b[k])
        tmp = sorted(tmp, reverse=True)
        v[i] = tmp
        k = k + 1

    return v


def to_tree(initial, relation):
    d = {}
    flag = [initial]
    q = []
    for i in relation[initial]:
        q.append(i)
        d[i] = 1

    while (len(q) != 0):
        front = q.pop(0)
        for j in relation[front]:
            if (d.get(j) == None or d[j] > d[front] + 1):
                d[j] = d[front] + 1
                q.append(j)
    layer = {}
    max_children = 0
    count = 0
    for i in relation.keys():
        if (i != initial and d[i] != inf):
            if (layer.get(d[i]) == None):
                layer[d[i]] = [i]
            else:
                layer[d[i]].append(i)
            edges = []
            for j in relation[i]:
                if (d[j] == d[i]):
                    relation[i].remove(j)
                    relation[j].remove(i)
                    count = count + 1
                if (d[i] == d[j] + 1):
                    edges.append(j)
            if (len(edges) != 0):
                residual = random.choice(edges)
            for j in edges:
                if (j != residual):
                    relation[i].remove(j)
                    relation[j].remove(i)
                    count = count + 1
    for i in layer.keys():
        tmp = 0
        if (i == 1):
            continue
        else:
            for j in layer[i]:
                if len(relation[j]) > 0:
                    tmp = tmp + 1
        if tmp > max_children:
            max_children = tmp
    return [relation, layer, max_children]


def ldm_tree(initial, supply, v, relation, layer, max_children):
    m = supply
    k = 1
    sw = 0
    revenue = 0
    while supply > 0:
        buyer = layer[k]
        next_layer = layer[k + 1]
        k = k + 1
        count = 0
        valuations = []
        if len(next_layer) > 0:
            for j in next_layer:
                if len(relation[j]) != 0:
                    next_layer.remove(j)
                    count = count + 1
        if len(next_layer) > 0:
            num = max_children + supply - count
            for j in next_layer:
                if (j != initial and len(v[j]) != 0):
                    valuations.append((v[j][0], j))
            valuations.sort(key=lambda x: x[0], reverse=True)
            if (len(valuations) > num):
                top = valuations[:num]
                for j in top:
                    next_layer.remove(j[1])
            else:
                next_layer = []
        buyer = buyer + next_layer
        valuations = []
        for j in buyer:
            if (j != initial):
                for n in v[j]:
                    valuations.append((n, j))
        valuations.sort(key=lambda x: x[0], reverse=True)
        top = valuations[:m]
        pwinner = {}
        for j in top:
            if (pwinner.get(j[1]) == None):
                pwinner[j[1]] = 1
            else:
                pwinner[j[1]] = pwinner[j[1]] + 1
        for j in pwinner:
            supply = supply - pwinner[j]
            sw = sw + sum(v[i][0:pwinner[j]])
            c = 0
            for n in range(pwinner[j]):
                if (len(valuations) > supply + n):
                    revenue = revenue + valuations[supply + n][0]
                    c = c + 1

    return [sw / m, revenue / m]


def mechanism(initial, priority, supply, v):
    buyer = relation[initial]
    accesible = copy.deepcopy(relation[initial])
    accesible.append(initial)
    distance = {}
    for i in range(nodes):
        distance[i] = inf
    for i in buyer:
        distance[i] = 1
    sw = 0
    revenue = 0
    winners = []
    m = supply
    while (m != 0):
        valuations = []
        buyer = list(set(buyer))
        accesible = list(set(accesible))
        for i in buyer:
            if (i != initial and len(v[i]) != 0):
                valuations.append((v[i][0], i))
        valuations.sort(key=lambda x: x[0], reverse=True)
        top = valuations[:m]
        pwinner = {}
        for i in top:
            if (pwinner.get(i[1]) == None):
                pwinner[i[1]] = 1
            else:
                pwinner[i[1]] = pwinner[i[1]] + 1
        tmp = []
        buyer = list(set(buyer))

        b = buyer.copy()
        for i in buyer:
            if (pwinner.get(i) == None):
                b.remove(i)
                if (relation.get(i) != None):
                    for j in relation[i]:
                        if (j != initial and (j in accesible) == False):
                            tmp.append(j)
                            accesible.append(j)
                            if (distance.get(j) == None or distance[j] > distance[i] + 1):
                                distance[j] = distance[i] + 1
        buyer = b
        buyer = buyer + tmp
        contribution = {}
        if (len(buyer) == 0):
            break
        if (len(pwinner) == len(buyer)):
            match priority:
                case 1:  # random
                    winner = random.choice(list(pwinner.keys()))

                case 2:  # new_agent
                    for i in pwinner.keys():
                        for j in relation[i]:
                            if (accesible.count(j) == 0):
                                if (contribution.get(i) == None):
                                    contribution[i] = 1
                                else:
                                    contribution[i] = contribution[i] + 1
                    if (len(contribution) != 0):
                        winner = max(contribution, key=contribution.get)
                    else:
                        if (len(pwinner) == 0):
                            break
                        winner = random.choice(list(pwinner))

                case 3:  # degree
                    for i in pwinner.keys():
                        contribution[i] = len(relation[i])
                    if (len(contribution) != 0):
                        winner = max(contribution, key=contribution.get)
                    else:
                        if (len(pwinner) == 0):
                            break
                        winner = random.choice(list(pwinner))
                case 4:  # depth
                    for i in pwinner.keys():
                        contribution[i] = distance[i]
                    if (len(contribution) != 0):
                        winner = max(contribution, key=contribution.get)
                    else:
                        if (len(pwinner) == 0):
                            break
                        winner = random.choice(list(pwinner))
                case 5:
                    for i in pwinner.keys():
                        contribution[i] = distance[i]
                    if (len(contribution) != 0):
                        winner = min(contribution, key=contribution.get)
                    else:
                        if (len(pwinner) == 0):
                            break
                        winner = random.choice(list(pwinner))

            winners.append((winner, pwinner[winner]))

            if (m >= len(valuations)):
                revenue = 0
            else:
                revenue = revenue + valuations[m][0]
            sw = sw + v[winner][0]
            del (v[winner][0])
            m = m - 1
            buyer.remove(winner)
            if (relation.get(winner) != None):
                for j in relation[winner]:
                    buyer.append(j)
                    accesible.append(j)
                    if (distance.get(j) == None or distance[j] > distance[winner] + 1):
                        distance[j] = distance[winner] + 1
    if (supply != m):
        return [len(accesible), sw / (supply - m), revenue / (supply - m)]
    else:
        return [len(accesible), 0, 0]


initials = random.sample(list(relation.keys()), 80)
# initials = list(relation.keys())
# initials=[0]

result_newagent_sw = []
result_newagent_revenue = []
result_degree_sw = []
result_degree_revenue = []
result_depth_sw = []
result_depth_revenue = []
result_random_sw = []
result_random_revenue = []
result_close_sw = []
result_close_revenue = []
result_optimal_sw = []
result_ldm_sw = []
result_ldm_revenue = []

for supply in range(60, 61):
    for i in initials:
        t1 = datetime.datetime.now().microsecond
        t2 = time.mktime(datetime.datetime.now().timetuple())
        newagent_sw = []
        newagent_revenue = []
        degree_sw = []
        degree_revenue = []
        depth_sw = []
        depth_revenue = []
        random_sw = []
        random_revenue = []
        close_sw = []
        close_revenue = []
        optimal_sw = []
        ldm_sw = []
        ldm_revenue = []
        tree = deepcopy(relation)
        tmp = to_tree(i, tree)
        tree = tmp[0]
        layer = tmp[1]
        max_children = tmp[2]

        for j in range(1):
            v = generate(i, supply, 2)
            # result = ldm_tree (i,supply,v,tree,layer,max_children)
            # ldm_sw.append(result[0])
            # ldm_revenue.append(result[1])
            # opt = []
            # for k in v.keys():
            #     for n in v[k]:
            #         opt.append(n)
            # opt = sorted(opt,reverse=True)
            # opt = opt[:supply]
            # opt_sw = sum(opt)
            # optimal_sw.append(opt_sw)
            for priority in range(1, 6):
                result = mechanism(i, priority, supply, v)
                match priority:
                    case 1:  # random

                        random_sw.append(result[1])
                        random_revenue.append(result[2])
                    case 2:  # new_agent

                        newagent_sw.append(result[1])
                        newagent_revenue.append(result[2])
                    case 3:  # degree

                        degree_sw.append(result[1])
                        degree_revenue.append(result[2])
                    case 4:  # depth

                        depth_sw.append(result[1])
                        depth_revenue.append(result[2])
                    case 5:

                        close_sw.append(result[1])
                        close_revenue.append(result[2])

        result_ldm_sw.append(np.nanmean(np.array(ldm_sw)))
        result_ldm_revenue.append(np.nanmean(np.array(ldm_revenue)))

        result_random_sw.append(np.nanmean(np.array(random_sw)))
        result_random_revenue.append(np.nanmean(np.array(random_revenue)))

        result_newagent_sw.append(np.nanmean(np.array(newagent_sw)))
        result_newagent_revenue.append(np.nanmean(np.array(newagent_revenue)))

        result_degree_sw.append(np.nanmean(np.array(degree_sw)))
        result_degree_revenue.append(np.nanmean(np.array(degree_revenue)))

        result_depth_sw.append(np.nanmean(np.array(depth_sw)))
        result_depth_revenue.append(np.nanmean(np.array(depth_revenue)))

        result_close_sw.append(np.nanmean(np.array(close_sw)))
        result_close_revenue.append(np.nanmean(np.array(close_revenue)))

        result_optimal_sw.append(np.nanmean(np.array(optimal_sw)))

        t3 = datetime.datetime.now().microsecond
        t4 = time.mktime(datetime.datetime.now().timetuple())
        strTime = 'funtion time use:%dms' % ((t4 - t2) * 1000 + (t3 - t1) / 1000)
        print(i, strTime)

average_result = []
average_result.append(result_newagent_sw)
average_result.append(result_newagent_revenue)
average_result.append(result_degree_sw)
average_result.append(result_degree_revenue)
average_result.append(result_depth_sw)
average_result.append(result_depth_revenue)
average_result.append(result_random_sw)
average_result.append(result_random_revenue)
average_result.append(result_close_sw)
average_result.append(result_close_revenue)
average_result.append(result_ldm_sw)
average_result.append(result_ldm_revenue)
average_result.append(result_optimal_sw)

average_result = np.array(average_result)
y1 = np.nanmean(average_result[0])
y2 = np.nanmean(average_result[2])
y3 = np.nanmean(average_result[4])
y4 = np.nanmean(average_result[6])
y5 = np.nanmean(average_result[8])
y6 = np.nanmean(average_result[1])
y7 = np.nanmean(average_result[3])
y8 = np.nanmean(average_result[5])
y9 = np.nanmean(average_result[7])
y10 = np.nanmean(average_result[9])

np.save('result_model2_60.npy', average_result)

average_newagent_sw = np.nanmean(np.array(result_newagent_sw))

average_newagent_revenue = np.nanmean(np.array(result_newagent_revenue))
average_degree_sw = np.nanmean(np.array(result_degree_sw))

average_degree_revenue = np.nanmean(np.array(result_degree_revenue))
average_depth_sw = np.nanmean(np.array(result_depth_sw))

average_depth_revenue = np.nanmean(np.array(result_depth_revenue))
average_random_sw = np.nanmean(np.array(result_random_sw))

average_random_revenue = np.nanmean(np.array(result_random_revenue))

average = []
for i in range(10):
    average.append(i)
