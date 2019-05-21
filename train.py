import torch
import argparse
from miner import Miner
from minesweeper import Minesweeper

parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=str, default='easy', help='choose from easy, middle and hard')
parser.add_argument('--epsilon', type=float, default=0.9, help='the probability to choose from memories')
parser.add_argument('--memory_capacity', type=int, default=10000, help='the capacity of memories')
parser.add_argument('--target_replace_iter', type=int, default=200, help='the iter to update the target net')
parser.add_argument('--batch_size', type=int, default=64, help='sample amount')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=10000, help='training epoch number')
parser.add_argument('--n_critic', type=int, default=500, help='evaluation point')
opt = parser.parse_args()
print(opt)

miner = Miner(opt.shape, opt.epsilon, opt.memory_capacity, opt.target_replace_iter, opt.batch_size, opt.lr)
print('collecting experience...')

win_num = 0
fail_num = 0
for epoch in range(opt.n_epochs):
    game = Minesweeper(opt.shape)
    s = game.get_state()
    critic_r = 0
    ep_r = 0
    while True:
        a = miner.choose_action(s)
        game.action(a)
        s_ = game.get_state()
        status = game.get_status()

        progress = s_ - s
        if status == 1:
            win_num += 1
            r = 1
        elif status == -1:
            fail_num += 1
            r = -1
        elif progress.sum() != 0:
            r = 0.9
        else:
            r = -0.3
        # print(s, s_, r, game.uncleared_blocks)

        miner.store_transition(s, a, r, s_)

        ep_r += r
        if miner.memory_counter > opt.memory_capacity:
            miner.learn()
            if game.get_status() != 0:
                print('Ep: ', epoch,
                      '| Ep_r: ', round(ep_r, 2))

        if status != 0:
            break

        s = s_.copy()

    critic_r += ep_r
    if (epoch+1) % opt.n_critic == 0:
        print('=====evaluation=====')
        print('Epochs:', epoch)
        print('win number:', win_num)
        print('fail number:', fail_num)
        print('win rate:', win_num / (win_num + fail_num))
        print('total reward:', critic_r)
        win_num = 0
        fail_num = 0
        critic_r = 0
