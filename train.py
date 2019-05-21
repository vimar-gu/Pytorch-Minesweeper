import torch
import argparse
from miner import Miner
from minesweeper import Minesweeper

parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=str, default='easy', help='choose from easy, middle and hard')
parser.add_argument('--epsilon', type=float, default=0.9, help='the probability to choose from memories')
parser.add_argument('--memory_capacity', type=int, default=2000, help='the capacity of memories')
parser.add_argument('--target_replace_iter', type=int, default=100, help='the iter to update the target net')
parser.add_argument('--batch_size', type=int, default=32, help='sample amount')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=1000, help='training epoch number')
parser.add_argument('--n_critic', type=int, default=200, help='evaluation point')
opt = parser.parse_args()
print(opt)

miner = Miner(opt.shape, opt.epsilon, opt.memory_capacity, opt.target_replace_iter, opt.batch_size, opt.lr)
print('collecting experience...')

win_num = 0
fail_num = 0
for epoch in range(opt.n_epochs):
    game = Minesweeper(opt.shape)
    s = game.get_state()
    score = game.get_score()
    ep_r = 0
    while True:
        a = miner.choose_action(s)
        game.action(a)
        s_ = game.get_state()
        score_ = game.get_score()
        status = game.get_status()

        r = score_ - score

        miner.store_transition(s, a, r, s_)

        ep_r += r
        if miner.memory_counter > opt.memory_capacity:
            miner.learn()
            if game.get_status() != 0:
                print('Ep: ', epoch,
                      '| Ep_r: ', round(ep_r, 2))

        if status == 1:
            win_num += 1
            ep_r += 10
            break
        elif status == -1:
            fail_num += 1
            ep_r -= 10
            break

        score = score_
        s = s_

    if (epoch+1) % opt.n_critic == 0:
        print('=====evaluation=====')
        print('Epochs:', epoch)
        print('win number:', win_num)
        print('fail number:', fail_num)
        print('win rate:', win_num / (win_num + fail_num))
        win_num = 0
        fail_num = 0
