# stores offset data and rotation matrices

class Data:
    def __init__(self, piece_id, test_rot_index):
        self.str_id = piece_id  # its a string
        self.rot_index = test_rot_index

    def O(self):
        O = {0: [(1, 1), (2, 1), (1, 2), (2, 2)],
             1: [(1, 1), (2, 1), (1, 2), (2, 2)],
             2: [(1, 1), (2, 1), (1, 2), (2, 2)],
             3: [(1, 1), (2, 1), (1, 2), (2, 2)]
             }
        return O[self.rot_index]

    def T(self):
        T = {0: [(0, 2), (1, 2), (1, 1), (2, 2)],
             1: [(1, 1), (1, 2), (1, 3), (2, 2)],
             2: [(0, 2), (1, 2), (2, 2), (1, 3)],
             3: [(1, 1), (1, 2), (1, 3), (0, 2)]
             }
        return T[self.rot_index]

    def I(self):
        I = {0: [(0, 1), (1, 1), (2, 1), (3, 1)],
             1: [(1, 0), (1, 1), (1, 2), (1, 3)],
             2: [(0, 1), (1, 1), (2, 1), (3, 1)],
             3: [(1, 0), (1, 1), (1, 2), (1, 3)]
             }
        return I[self.rot_index]

    def L(self):
        L = {0: [(1, 2), (2, 2), (3, 2), (3, 1)],
             1: [(2, 1), (2, 2), (2, 3), (3, 3)],
             2: [(1, 2), (2, 2), (1, 3), (3, 2)],
             3: [(1, 1), (2, 2), (2, 1), (2, 3)]
             }
        return L[self.rot_index]

    def Z(self):
        Z = {0: [(0, 1), (1, 2), (1, 1), (2, 2)],
             1: [(2, 1), (1, 2), (2, 2), (1, 3)],
             2: [(0, 2), (1, 2), (1, 3), (2, 3)],
             3: [(0, 2), (1, 2), (1, 1), (0, 3)]
             }
        return Z[self.rot_index]

    def J(self):
        J = {0: [(0, 1), (1, 2), (0, 2), (2, 2)],
             1: [(1, 1), (1, 2), (1, 3), (2, 1)],
             2: [(0, 2), (1, 2), (2, 2), (2, 3)],
             3: [(1, 1), (1, 2), (1, 3), (0, 3)]
             }
        return J[self.rot_index]

    def S(self):
        S = {0: [(0, 2), (1, 2), (1, 1), (2, 1)],
             1: [(1, 1), (1, 2), (2, 2), (2, 3)],
             2: [(1, 1), (1, 2), (2, 2), (2, 3)],
             3: [(1, 3), (1, 2), (0, 3), (2, 2)]
             }
        return S[self.rot_index]

    def get_data(self):
        if self.str_id == 'O':
            return self.O()
        elif self.str_id == 'I':
            return self.I()
        elif self.str_id == 'Z':
            return self.Z()
        elif self.str_id == 'S':
            return self.S()
        elif self.str_id == 'T':
            return self.T()
        elif self.str_id == 'L':
            return self.L()
        elif self.str_id == 'J':
            return self.J()
