

import numpy as np




class sequence(object):

    def __init__(self, sequence_len=10):

        self.sequence_len = sequence_len

        self.grid_width = 28
        self.grid_height = self.grid_width
        self.block_size = 4
        self.block_speed = 5



    def get_batch(self, batch_size=5):

        batch = []
        for i in range(batch_size):
            sequence = []
            pos_x = 0
            pos_y = 0
            for t in range(self.sequence_len):
                grid = np.zeros([self.grid_width, self.grid_height])
                grid[pos_x:pos_x+self.block_size, pos_y:pos_y+self.block_size,] = 1.
                sequence.append(grid)

                pos_x += self.block_speed
                pos_y += self.block_speed

                if pos_x > self.grid_width:
                    pos_x, pos_y = [0,0]


            batch.append(sequence)





        #[B,T,X]
        return batch















