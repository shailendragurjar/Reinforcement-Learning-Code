#!/usr/bin/env python

# bandit-agent.py
# -------------

import math
import numpy as np
import time
import socket
from optparse import OptionParser
import errno

debug = True


def opts():
    p = OptionParser()
    seed = int(time.time() * 1000)
    p.add_option("-n", "--numArms"   , type = int  , default = 5)
    p.add_option("-r", "--randomSeed", type = int  , default = seed)
    p.add_option("-z", "--horizon"   , type = int  , default = 200)
    p.add_option("-s", "--hostname"  , type = str  , default = "localhost")
    p.add_option("-p", "--port"      , type = int  , default = "5000")
    p.add_option("-a", "--algorithm" , type = str  , default = "random")
    p.add_option("-e", "--epsilon"   , type = float, default = 0.0)
    return p


def send(sock, msg):
    totalsent = 0
    msg = msg + '\0'  # add null so C++ can find end of string
    msglen = len(msg)
    while totalsent < msglen:
        sent = sock.send(msg[totalsent:])
        if sent == 0:
            raise RuntimeError("send: socket connection broken")
        totalsent = totalsent + sent
    if debug:
        print 'sent %d bytes, hex %s, text %s' % (totalsent,
                                                  msg.encode('hex'),
                                                  msg)
    return totalsent


def recv(sock, bufsize=256):
    try:
        chunk = sock.recv(bufsize)

        if chunk == '':
            # raise RuntimeError("recv: socket connection broken")
            if debug:
                print '### recv: socket connection broken'
            return None

        if debug:
            print '### received %s bytes, hex %s, text %s' % (
                    len(chunk),
                    chunk.encode('hex'),
                    chunk)

        return chunk.strip(b'\0')
    except socket.error as e:
        if e.errno == errno.ECONNRESET:
            # this happends when server closes the connection,
            # after all horizons are done
            if debug:
                print '### recv: ', e
            return None
        else:
            raise e


def getreward(sock, arm_to_pull):
    print 'Sending action %s' % arm_to_pull
    send(sock, str(arm_to_pull))

    recvd = recv(sock)
    if not recvd:
        return None, None

    tmp = recvd.split(',')  # server reply , separated

    reward = float(tmp[0])
    pulls = int(tmp[1])
    if debug:
        print '### reward %f, pulls %d' % (reward, pulls)

    return reward, pulls


def update_estimate(K, est_values, reward, arm_to_pull):
    K[arm_to_pull] += 1
    alpha = 1./K[arm_to_pull]
    est_values[arm_to_pull] += alpha * (reward - est_values[arm_to_pull])


def rr(options, sock):
    K = np.zeros(options.numArms)
    est_values = np.zeros(options.numArms)

    pulls = 0
    reward = 0.0

    while True:
        # pick an arm
        arm_to_pull = pulls % options.numArms
        reward, pulls = getreward(sock, arm_to_pull)
        if reward is None:
            break
        # update_estimate
        update_estimate(K, est_values, reward, arm_to_pull)


def epsilon_greedy(options, sock):
    K = np.zeros(options.numArms)
    est_values = np.zeros(options.numArms)

    pulls = 0
    reward = 0.0

    while True:
        # pick an arm
        rand_num = np.random.random()
        if options.epsilon > rand_num:
            arm_to_pull = np.random.randint(options.numArms)
        else:
            arm_to_pull = np.argmax(est_values)

        reward, pulls = getreward(sock, arm_to_pull)

        if reward is None:
            break

        # update_estimate
        update_estimate(K, est_values, reward, arm_to_pull)


def ucb(options, sock):
    K = np.zeros(options.numArms)
    est_values = np.zeros(options.numArms)

    pulls = 0
    reward = 0.0

    for i in range(options.numArms):
        arm_to_pull = i

        reward, pulls = getreward(sock, arm_to_pull)

        if reward is None:
            raise 'server connection broken too early, after', pulls, 'pulls'

        update_estimate(K, est_values, reward, arm_to_pull)

    while True:
        # pick an arm
        arm_to_pull = np.argmax(est_values +
                                np.sqrt(np.reciprocal(K) * 2 * np.log(pulls)))

        reward, pulls = getreward(sock, arm_to_pull)

        if reward is None:
            break

        update_estimate(K, est_values, reward, arm_to_pull)


def kl_ucb(options, sock):
    K = np.zeros(options.numArms)
    est_values = np.zeros(options.numArms)

    pulls = 0
    reward = 0.0

    for i in range(options.numArms):
        arm_to_pull = i

        reward, pulls = getreward(sock, arm_to_pull)

        if reward is None:
            raise 'server connection broken too early, after', pulls, 'pulls'

        update_estimate(K, est_values, reward, arm_to_pull)

    est_values = est_values + 1e-5

    if debug:
        print '###kl_ucb: est_values', est_values

    while True:
        # pick an arm
        arm_to_pull = np.argmax(np.array(
                        map(lambda k, est: pick_arm_kl_ucb(k, est, pulls),
                            K, est_values)))

        reward, pulls = getreward(sock, arm_to_pull)

        if reward is None:
            break

        update_estimate(K, est_values, reward, arm_to_pull)


def pick_arm_kl_ucb(k, est, pulls):
    if debug:
        print '###pick_arm_kl_ucb', k, est, pulls

    C = 1.0  # const
    lowerbound = -float('inf')
    precision = 1e-6

    d = C * math.log(pulls) / k
    upperbound = min(1., est + math.sqrt(2 * d))

    l = max(est, lowerbound)
    u = upperbound

    if debug:
        print '###pick_arm_kl_ucb: l', l, 'u', u

    while u-l > precision:
        m = (l+u)/2.0

        if (est * math.log(est / m) +
                (1 - est) * math.log((1 - est) / (1 - m))) > d:
            u = m
        else:
            l = m

    return (l + u) / 2


def thompson(options, sock):
    success = np.zeros(options.numArms)
    failure = np.zeros(options.numArms)

    pulls = 0
    reward = 0.0

    while True:
        # pick an arm
        arm_to_pull = np.argmax(np.random.beta(success + 1, failure + 1))

        reward, pulls = getreward(sock, arm_to_pull)

        if reward is None:
            break

        if reward > 0:
            success[arm_to_pull] += 1
        else:
            failure[arm_to_pull] += 1


##############################################################################
# MAIN
##############################################################################
(options, args) = opts().parse_args()

if debug:
    print '###', options, args

np.random.seed(options.randomSeed)  # seed
# np.random.seed(47)  # setting const value will give reproducable resutls

remotehost = socket.gethostbyname(options.hostname)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    if debug:
        print '### connecting to', remotehost, options.port

    s.connect((remotehost, options.port))

    if options.algorithm == "rr":
        rr(options, s)
    elif options.algorithm == "epsilon-greedy":
        epsilon_greedy(options, s)
    elif options.algorithm == "UCB":
        ucb(options, s)
    elif options.algorithm == "KL-UCB":
        kl_ucb(options, s)
    elif options.algorithm == "Thompson-Sampling":
        thompson(options, s)
    else:
        raise "Algorithm not implemented: " + options.algorithm

finally:
    s.close()
