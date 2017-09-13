def train_faster_rcnn():
    from faster_rcnn.solver import Solver
    frcnn_solver = Solver()
    frcnn_solver.train()

if __name__ == '__main__':
    train_faster_rcnn()