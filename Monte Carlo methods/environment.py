import random

class Environment():
    def __init__(self, filename='racetrack1_32x17.txt'):    
        self.scenario = self._load_scenario(filename)
        self.finish = "f"
        self.start = "s"
        self.track = " "
        self.wall = "+"
        self.waypoint = "·"
        self.path = []
    
    def _load_scenario(self, filename):
        with open(filename) as f:
            scenario = list(map(lambda line: [c for c in line], f.read().split("\n")))
        return scenario
    
    def print(self, car):
        top_nums = "  " + "".join(map(lambda n: str(n%10), range(0, len(self.scenario[0]))))
        buffer = top_nums + "\n  " + "-"*len(self.scenario[0])
        buffer += "\n"
        
        for row_index, row in enumerate(self.scenario):
            buffer += str(row_index % 10) + "|"
            for col_index, col in enumerate(self.scenario[row_index]):
                if car.position == (row_index, col_index):
                    buffer += str(car)
                    continue
                if any(filter(lambda waypoint: waypoint == (row_index, col_index), self.path)):
                    buffer += self.waypoint
                    continue
                
                buffer += self.scenario[row_index][col_index]
            
            buffer += "\n"
            
        print(buffer)
    
    def select_start_position(self):
        starts = []
        for row_index, row in enumerate(self.scenario):
            for col_index, col in enumerate(self.scenario[row_index]):
                if self.scenario[row_index][col_index] == self.start:
                    starts.append((row_index, col_index))
        return random.choice(starts)
    
    def move_to(self, car, new_position):
        position = car.position
        increment_v = 1 if new_position[0] - position[0] >= 0 else -1
        increment_h = 1 if new_position[1] - position[1] >= 0 else -1
        
        path = [position]
        while position != new_position:
            if abs(new_position[0] - position[0]) >= abs(new_position[1] - position[1]):
                position = (position[0] + increment_v, position[1])
            else:
                position = (position[0], position[1] + increment_h)
                
            path.append(position)
            if self.is_wall(position):
                return self.select_start_position(), path
            elif self.is_finish(position):
                return position, path
                
        return new_position, path
    
    def is_track(self, position):
        if self.scenario[position[0]][position[1]] == self.track:
            return True
        
        return False
    
    def is_wall(self, position):
        if position[0] < 0 or position[0] > len(self.scenario)-1:
            return True
        elif position[1] < 0 or position[1] > len(self.scenario[position[0]])-1:
            return True
        elif self.scenario[position[0]][position[1]] == self.wall:
            return True
        else:
            return False
    
    def is_start(self, position):
        if self.scenario[position[0]][position[1]] == self.start:
            return True
        return False
    
    def is_finish(self, position):
        if self.scenario[position[0]][position[1]] == self.finish:
            return True
        return False


env = Environment("racetrack2_30x32.txt")