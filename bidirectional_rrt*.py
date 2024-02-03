

import random
import numpy as np
from typing import Sequence, List
import matplotlib.pyplot as plt
import matplotlib.lines as line
from matplotlib.collections import LineCollection

from dg_commons import SE2Transform
import shapely
import time

# start_time = time.time()


class Node:
    def __init__(self, p: SE2Transform):
        self.state = p
        self.parent = None
        self.idx = None
        self.cost = None

        self.left = None
        self.right = None
        # k-d tree children for nearest neighbour search

    def set_parent(self, parent_node):
        self.parent = parent_node

    def set_idx(self, idx):
        self.idx = idx

    def set_certificate(self, r):
        self.certificate = r

    def set_cost(self, c):
        self.cost = c

    def distance(self, node):
        p1 = self.state
        p2 = node.state
        return euclid_dist(p1, p2)

    def calc_heading(self, node):
        p1 = node.state.p
        p2 = self.state.p

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        self.state.theta = np.rad2deg(np.arctan(dy / dx))


def build_kd_tree(nodes: List[Node], depth=0):
    '''
    This function builds a kd-tree to accelerate nearest node search

    parameters:
    nodes: List of nodes to be split up
    depth: used for recursion
    '''
    if not nodes:
        return None

    k = len(nodes[0].state.p)
    axis = depth % k
    sorted_points = sorted(nodes, key=lambda x: x.state.p[axis])
    median_index = len(sorted_points) // 2
    median_node = sorted_points[median_index]

    median_node.left = build_kd_tree(
        [n for n in nodes if n.state.p[axis] < median_node.state.p[axis]], depth + 1
    )
    median_node.right = build_kd_tree(
        [n for n in nodes if n.state.p[axis] > median_node.state.p[axis]], depth + 1
    )

    return median_node


class RRT_star:
    '''
    Bi-directional RRT* 

    Two trees, one that grows 
    from the start and another that grows from the goal

    main attributes:

    step_size: max distance between nodes
    goal_bias: Ratio , choose s.t 1/goal_bias is whole
    proximity_radius: optimization parameter, usally set equal to step_size
    balance_ratio: Frequency of kd-tree rebalancing (1/balance_ratio must be whole)
    num_iter: number of itterations per search cycle

    '''
    def __init__(
        self,
        start: SE2Transform,
        goal: SE2Transform,
        bounds,
        obstacle_list: List,
        step_size=0.5,
        goal_bias=0.5,
        proximity_radius=0.5,
        robot_radius=1,
        balance_ratio=0.1,
        num_iter=100,
    ):
        self.start_node = Node(start)
        self.start_node.set_idx(0)
        self.goal_node = Node(goal)
        self.goal_start_node = Node(goal)
        self.goal_start_node.set_idx(0)
        self.start_goal = Node(start)
        self.robot_radius = robot_radius

        # Graph parameters
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.proximity_radius = proximity_radius

        self.vertices = [self.start_node]
        self.distances = {0: 0.0}

        self.kd_tree = None
        self.edges = []

        #Goal graph parameters
        self.g_vertices = [self.goal_start_node]
        self.g_distances = {0: 0.0}
        self.g_kd_tree = None
        self.g_edges = []

        #Output
        self.Path = []

        # Algorithm parameters
        self.num_iter = num_iter
        self.balance_ratio = balance_ratio
        self.itt_step = 0

        # Obstacles
        self.bound_x = (bounds[0], bounds[2])
        self.bound_y = (bounds[1], bounds[3])
        self.obstacles = obstacle_list

    def add_vertice(self, pos: Node,s_g):
        if s_g == 0:
            if pos in self.vertices:
                idx = self.vertices.index(pos)
            else:
                self.vertices.append(pos)
                idx = len(self.vertices) - 1
                pos.set_idx(idx)
            return idx
        if s_g == 1:
            if pos in self.g_vertices:
                idx = self.g_vertices.index(pos)
            else:
                self.g_vertices.append(pos)
                idx = len(self.g_vertices) - 1
                pos.set_idx(idx)
            return idx
        
    def add_edge(self, idx1, idx2,s_g):
        if s_g == 0:
            self.edges.append((idx1, idx2))
        if s_g == 1:
            self.g_edges.append((idx1, idx2))

    def new_rand_pos(self):
        """
        Randomly assigns next pos (with occasional bias)
        """

        rx, ry = random.random(), random.random()

        max_dist_x = self.bound_x[1]-self.bound_x[0]
        max_dist_y = self.bound_y[1]-self.bound_y[0]
      
        pos_x = self.bound_x[0]+rx*max_dist_x
        pos_y = self.bound_y[0]+ry*max_dist_y

        return pos_x, pos_y

    def obstacle_collision(self, n: Node, near_n: Node):
        p1 = n.state.p
        p2 = near_n.state.p
        line = shapely.geometry.LineString([p1, p2])
        for obs in self.obstacles:
            coords = obs.exterior.coords
            if line.intersects(obs):
                return True
        return False
    
    def optimizer_1(self,min_node: Node,min_distance,new_node: Node,idx_new,s_g):

        if s_g == 0:
            for v in self.vertices:
                if v == new_node:
                    continue
                # Find the distance between v and the newest node
                dist_to_new = v.distance(new_node)

                # If dist smaller than proximity_radius, potential for improvement is high
                if (
                    dist_to_new > self.proximity_radius
                    or dist_to_new > self.step_size - 1e-6
                ):
                    continue

                # If the path between v and new_node is blocked, continue
                if self.obstacle_collision(v, new_node):
                    continue

                # If we made it to here, it means there is potential
                # for a better path to be created through node v, rather
                # than from the nearest node
                cost = self.distances[v.idx] + dist_to_new

                if cost < self.distances[idx_new]:
                    min_distance = dist_to_new
                    self.distances[idx_new] = cost
                    min_node = v
        if s_g == 1:
            for v in self.g_vertices:
                if v == new_node:
                    continue
                # Find the distance between v and the newest node
                dist_to_new = v.distance(new_node)

                # If dist smaller than proximity_radius, potential for improvement is high
                if (
                    dist_to_new > self.proximity_radius
                    or dist_to_new > self.step_size - 1e-6
                ):
                    continue

                # If the path between v and new_node is blocked, continue
                if self.obstacle_collision(v, new_node):
                    continue

                # If we made it to here, it means there is potential
                # for a better path to be created through node v, rather
                # than from the nearest node
                cost = self.g_distances[v.idx] + dist_to_new

                if cost < self.g_distances[idx_new]:
                    min_distance = dist_to_new
                    self.g_distances[idx_new] = cost
                    min_node = v

        return min_distance,min_node

    def optimizer_2(self,new_node: Node,idx_new,s_g):
        if s_g == 0:
            for v in self.vertices:
                if v == new_node:
                    continue

                # Find the distance between v and the newest node
                dist_to_new = v.distance(new_node)

                # If dist smaller than proximity_radius, potential for improvement is high
                if dist_to_new > self.proximity_radius or dist_to_new > self.step_size:
                    continue

                cost = self.distances[idx_new] + dist_to_new
                dist_to_v = self.distances[v.idx]

                if cost < dist_to_v:
                    if self.obstacle_collision(v, new_node):
                        continue
                    parent_node = v.parent
                    v.set_parent(new_node)
                    edge_to_cut = (parent_node.idx, v.idx)
                    self.edges.remove(edge_to_cut)
                    self.add_edge(idx_new, v.idx,0)

        if s_g == 1:
            for v in self.g_vertices:
                if v == new_node:
                    continue

                # Find the distance between v and the newest node
                dist_to_new = v.distance(new_node)

                # If dist smaller than proximity_radius, potential for improvement is high
                if dist_to_new > self.proximity_radius or dist_to_new > self.step_size:
                    continue

                cost = self.g_distances[idx_new] + dist_to_new
                dist_to_v = self.g_distances[v.idx]

                if cost < dist_to_v:
                    if self.obstacle_collision(v, new_node):
                        continue
                    parent_node = v.parent
                    v.set_parent(new_node)
                    edge_to_cut = (parent_node.idx, v.idx)
                    self.g_edges.remove(edge_to_cut)
                    self.add_edge(idx_new, v.idx,1)

    def iterate(self):
        """
        Method that builds the RRT*
        """
        goal_found = False
        for i in range(self.num_iter):

            if self.goal_bias != 0:
                if i % (1 / self.goal_bias) == 0 and goal_found == False:
                    rand_pos_x, rand_pos_y = self.goal_node.state.p[0], self.goal_node.state.p[1]
                    g_rand_pos_x, g_rand_pos_y = self.start_node.state.p[0], self.start_node.state.p[1]
                else:
                    rand_pos_x, rand_pos_y = self.new_rand_pos()
                    g_rand_pos_x, g_rand_pos_y = self.new_rand_pos()
                
            else:
                rand_pos_x, rand_pos_y = self.new_rand_pos()
                g_rand_pos_x, g_rand_pos_y = self.new_rand_pos()

            if rand_pos_x < self.bound_x[0] or rand_pos_x > self.bound_x[1]:
                continue
            if rand_pos_y < self.bound_y[0] or rand_pos_y > self.bound_y[1]:
                continue
            if g_rand_pos_x < self.bound_x[0] or g_rand_pos_x > self.bound_x[1]:
                continue
            if g_rand_pos_y < self.bound_y[0] or g_rand_pos_y > self.bound_y[1]:
                continue

            rand_pos = SE2Transform([rand_pos_x, rand_pos_y], 0)
            g_rand_pos = SE2Transform([g_rand_pos_x, g_rand_pos_y], 0)
            new_rand_node = Node(rand_pos)
            new_g_rand_node = Node(g_rand_pos)

            # Building the k-d tree for nearest neighbour search
            # The tree must be updadted, so it stays balanced --> How often should it be updated? Set at 10 rn.
            if i <= 10 or i % (1 / self.balance_ratio) == 0:
                self.kd_tree = build_kd_tree(self.vertices)
                self.g_kd_tree = build_kd_tree(self.g_vertices)

            # Find nearest:
            nearest_node = nearest(self.kd_tree, new_rand_node)
            g_nearest_node = nearest(self.g_kd_tree, new_g_rand_node)

            nearest_idx = nearest_node.idx
            g_nearest_idx = g_nearest_node.idx

            min_node = nearest_node
            g_min_node = g_nearest_node

            # Add new vertice:
            # Collision check needed here to make sure the path is clear of obstacles
            new_node = new_vertice(new_rand_node, nearest_node, self.step_size)
            g_new_node = new_vertice(new_g_rand_node,g_nearest_node,self.step_size)
            if (
                new_node.state.p[0] == self.goal_node.state.p[0]
                and new_node.state.p[1] == self.goal_node.state.p[1]
            ):
                goal_found = True
                continue

            if (
                g_new_node.state.p[0] == self.start_node.state.p[0]
                and g_new_node.state.p[1] == self.start_node.state.p[1]
            ):
                goal_found = True
                continue

            # Check for collision:
            # A new node is only added to tree if doesn't
            # collide with an obstacle
            if self.obstacle_collision(new_node, nearest_node):
                i-=1
                continue
            if self.obstacle_collision(g_new_node, g_nearest_node):
                i-=1
                continue

            idx_new = self.add_vertice(new_node,0)
            g_idx_new = self.add_vertice(g_new_node,1)

            # Add new edge candidate:
            min_distance = nearest_node.distance(new_node)
            g_min_distance = g_nearest_node.distance(g_new_node)

            self.distances[idx_new] = self.distances[nearest_idx] + min_distance
            self.g_distances[g_idx_new] = self.g_distances[g_nearest_idx] + g_min_distance

            # Update the edges to find a more optimal path
            min_distance, min_node = self.optimizer_1(min_node,min_distance,new_node,idx_new,0)
            g_min_distance,g_min_node = self.optimizer_1(g_min_node,g_min_distance,g_new_node,g_idx_new,1)

            #Set parent of new node:
            self.add_edge(min_node.idx, idx_new,0)
            new_node.set_parent(min_node)

            self.add_edge(g_min_node.idx, g_idx_new,1)
            g_new_node.set_parent(g_min_node)
    
            # Remove all non-optimal edges and replace with new optimal edge
            self.optimizer_2(new_node,idx_new,0)
            self.optimizer_2(g_new_node,g_idx_new,1)

            #-------- END OF ITTERATION ---------

        #We first check if the start tree has found the goal
        self.kd_tree = build_kd_tree(self.vertices)
        nearest_to_goal = nearest(self.kd_tree, self.goal_node)
        print(nearest_to_goal.state.p)
        dist_to_end = nearest_to_goal.distance(self.goal_node)

        if self.termination_test(dist_to_end):
            if not (
            self.obstacle_collision(self.goal_node, nearest_to_goal)
            ) :
                # If the goal is found for the first time
                if self.goal_node.parent is None:
                    self.goal_node.set_parent(nearest_to_goal)
                    idx_goal = self.add_vertice(self.goal_node,0)
                    self.add_edge(nearest_to_goal.idx, idx_goal,0)
                    self.distances[idx_goal] = (
                        self.distances[nearest_to_goal.idx] + dist_to_end
                    )
                # Only changes the goal parent if it provides an imporvement
                else:
                    cost = self.distances[nearest_to_goal.idx] + dist_to_end
                    if cost < self.distances[self.goal_node.idx]:
                        self.goal_node.set_parent(nearest_to_goal)
                        self.add_edge(nearest_to_goal.idx, self.goal_node.idx,0)
                    self.distances[self.goal_node.idx] = (
                        self.distances[nearest_to_goal.idx] + dist_to_end
                    )
                print("goal found from start!")
                self.Path = self.shortest_path([], self.goal_node)

        #If start tree still hasn't found goal, maybe the goal tree has.
        if len(self.Path) == 0:
            print("maybe from end")
            self.g_kd_tree = build_kd_tree(self.g_vertices)
            nearest_to_start = nearest(self.g_kd_tree, self.start_goal)
            print("from goal tree",nearest_to_start.state.p)
            g_dist_to_start = nearest_to_start.distance(self.start_goal)
            if self.termination_test(g_dist_to_start):
                if not (
                self.obstacle_collision(self.start_goal, nearest_to_start)
                ) :
                    # If the goal is found for the first time
                    if self.start_goal.parent is None:
                        self.start_goal.set_parent(nearest_to_start)
                        g_idx_start = self.add_vertice(self.start_goal,1)
                        self.add_edge(nearest_to_start.idx, g_idx_start,1)
                        self.g_distances[g_idx_start] = (
                            self.g_distances[nearest_to_start.idx] + g_dist_to_start
                        )
                    # Only changes the goal parent if it provides an imporvement
                    else:
                        cost = self.g_distances[nearest_to_start.idx] + g_dist_to_start
                        if cost < self.g_distances[self.start_goal.idx]:
                            self.start_goal.set_parent(nearest_to_start)
                            self.add_edge(nearest_to_start.idx, self.start_goal.idx,1)
                        self.g_distances[self.start_goal.idx] = (
                            self.g_distances[nearest_to_start.idx] + dist_to_end
                        )
                    print("goal found from end!")
                    self.Path = self.shortest_path([], self.start_goal)
                    self.Path.reverse()

        #If neither has found goal maybe they can be connected together!
        if len(self.Path) == 0:
            contact = False
            best_contact = [0,0]
            for v in self.vertices:
                dist_to_goal = 1e6
                nearest_g_node = nearest(self.g_kd_tree,v)
                dist_to_v = v.distance(nearest_g_node)

                # Test if trees are "touching"
                if dist_to_v < self.step_size:
                    if not(self.obstacle_collision(v,nearest_g_node)):
                        contact = True
                        d_g = self.distances[v.idx] + self.g_distances[nearest_g_node.idx]
                        if d_g < dist_to_goal:
                            dist_to_goal = d_g
                            best_contact = [v.idx,nearest_g_node.idx]
            if contact:
                print("found fusion")
                #Find path goal to nearest:
                g_node = self.g_vertices[best_contact[1]]
                g_Path = [g_node]
                while g_node.parent is not None:
                    g_node = g_node.parent
                    g_Path.append(g_node)
                self.Path = self.shortest_path([],self.vertices[best_contact[0]])
                g_Path.reverse()
                self.Path = g_Path + self.Path
            else: 
                print("no fusion yet")

    def termination_test(self, dist):
        """
        Method to test if new node collides with end goal object or is next to it
        """
        # Change this
        if dist < (self.step_size + self.robot_radius):
            return True
        else:
            return False

    def shortest_path(self, Path: List[Node], node: Node):
        Path.append(node)
        while node.parent is not None:
            node = node.parent
            Path.append(node)
        return Path

    def test_print(self,ax):
        nodes_x = []
        nodes_y = []

        g_nodes_x = []
        g_nodes_y = []

        path_x = []
        path_y = []
        lineN = []
        g_lineN = []


    
        for i, v in enumerate(self.vertices):
            nodes_x.append(v.state.p[0])
            nodes_y.append(v.state.p[1])

        for i, v in enumerate(self.g_vertices):
            g_nodes_x.append(v.state.p[0])
            g_nodes_y.append(v.state.p[1])
        
        for v in self.Path:
            path_x.append(v.state.p[0])
            path_y.append(v.state.p[1])

        for e in self.edges:
            edge1 = self.vertices[e[0]].state.p
            edge2 = self.vertices[e[1]].state.p
            lineN.append([edge1,edge2])
        
        for e in self.g_edges:
            edge1 = self.g_vertices[e[0]].state.p
            edge2 = self.g_vertices[e[1]].state.p
            g_lineN.append([edge1,edge2])

        ax.set_xlim(-11,11)
        ax.set_ylim(-11,11)

        line_segments1 = LineCollection(lineN,
                               color="red", linestyle='solid',linewidth=0.5)
        line_segments2 = LineCollection(g_lineN,
                               color="magenta", linestyle='solid',linewidth=0.5)
        ax.add_collection(line_segments1)
        ax.add_collection(line_segments2)
        ax.scatter(nodes_x, nodes_y, 3,color='green')
        ax.scatter(g_nodes_x, g_nodes_y, 3,color='black')
        ax.plot(path_x, path_y, marker="o", linestyle="-", markersize=1)


def nearest(root_node: Node, node: Node):
    """
    Finds the nearest node
    parameters:
    root_node: root of the tree
    node: node under test
    """
    # Distances
    dist_root = float("inf")
    dist_left = float("inf")
    dist_right = float("inf")

    # Distance to root node
    dist_root = euclid_dist(root_node.state, node.state)

    # children
    nl = root_node.left
    nr = root_node.right
    if nl is not None:
        dist_left = euclid_dist(nl.state, node.state)
    if nr is not None:
        dist_right = euclid_dist(nr.state, node.state)

    # Find minimum:
    if dist_root < dist_left and dist_root < dist_right:
        return root_node
    if dist_left < dist_right:
        return nearest(nl, node)
    else:
        return nearest(nr, node)


def new_vertice(rand_vert: Node, near_vert: Node, step_size):
    """This function computes the new random node on the RRT* tree that is within one step
    size of the nearest existing node
    """
    rand_pos = rand_vert.state.p
    near_pos = near_vert.state.p

    # Directional vector
    dir_vec = np.array(rand_pos) - np.array(near_pos)
    l = np.linalg.norm(dir_vec)

    if l > 0:
        dir_vec = (dir_vec / l) * min(step_size, l)
        new_pos = SE2Transform([near_pos[0] + dir_vec[0], near_pos[1] + dir_vec[1]], 0)
        return Node(new_pos)
    else:
        return near_vert


def euclid_dist(p1: SE2Transform, p2: SE2Transform):
    d = ((p1.p[0] - p2.p[0]) ** 2 + (p1.p[1] - p2.p[1]) ** 2) ** (1 / 2)
    return d



if __name__ == "__main__":
    t0 = time.time()
    start = SE2Transform([-2.5, -10], 1)
    end = SE2Transform([10,10 ], 0)
    fig, ax = plt.subplots()

    coords1 = ((-1, -7), (-1, -3), (3, -3), (3,-7), (-1,-7))
    coords2 = ((5,-7), (5,-3), (9,-3), (9,-7), (5,-7))
    coords3 = ((-1,-1), (-1,3), (3,3), (3,-1), (-1,-1))
    coords4 = ((5,-1), (5,3), (9,3), (9,-1), (5,-1))
    coords5 = ((-6,5), (-5,6), (-4,6), (-5,4), (-6,5))
    coords6 = ((6,5), (7,7), (8,6), (7,4), (6,5))
    coords7 = ((-7,-7),(-7,-3),(-3,-3), (-3,-7), (-7,-7))
    coords8 = ((-1,-7),(-1,-3),(3,-3), (3,-7), (-1,-7))
    coords9 = ((0,5),(1,7),(2,6), (1,4), (0,5))
    coords10 = ((-7,-1),(-7,3),(-3,3), (-3,-1), (-7,-1))
    coords11 = ((2.5,2),(2.5,4),(5,4), (5,2), (2.5,2))
    cx1, cy1 = [-1, -1, 3, 3], [-7, -3, -3, -7]
    cx2, cy2 = [5,5,9,9], [-7,-3,-3,-7]
    cx3, cy3 = [-1,-1,3,3], [-1,3,3,-1]
    cx4, cy4 = [5,5,9,9], [-1,3,3,-1]
    cx5, cy5 = [-6,-5,-4,-5], [5,6,6,4]
    cx6, cy6 = [6,7,8,7], [5,7,6,4]
    cx7, cy7 = [-7,-7,-3,-3], [-7,-3,-3,-7]
    cx8, cy8 = [-1,-1,3,3], [-7,-3,-3,-7]
    cx9, cy9 = [0,1,2,1], [5,7,6,4]
    cx10, cy10 = [-7,-7,-3,-3],[-1,3,3,-1]
    cx11, cy11 = [2.5,2.5,5,5],[2,4,4,2]
    ax.fill(cx1, cy1)
    ax.fill(cx2, cy2)
    ax.fill(cx3, cy3)
    ax.fill(cx4, cy4)
    ax.fill(cx5, cy5)
    ax.fill(cx6, cy6)
    ax.fill(cx7, cy7)
    ax.fill(cx8, cy8)
    ax.fill(cx9, cy9)
    ax.fill(cx10, cy10)
    ax.fill(cx11, cy11)
    polygon1 = shapely.geometry.Polygon(coords1).buffer(0.6)
    polygon2 = shapely.geometry.Polygon(coords2).buffer(0.6)
    polygon3 = shapely.geometry.Polygon(coords3).buffer(0.6)
    polygon4 = shapely.geometry.Polygon(coords4).buffer(0.6)
    polygon5 = shapely.geometry.Polygon(coords5).buffer(0.6)
    polygon6 = shapely.geometry.Polygon(coords6).buffer(0.6)
    polygon7 = shapely.geometry.Polygon(coords7).buffer(0.6)
    polygon8 = shapely.geometry.Polygon(coords8).buffer(0.6)
    polygon9 = shapely.geometry.Polygon(coords9).buffer(0.6)
    polygon10 = shapely.geometry.Polygon(coords10).buffer(0.6)
    polygon11 = shapely.geometry.Polygon(coords11)

    
    G1 = RRT_star(
        start,
        end,
        bounds = [-11,-11,11,11],
        obstacle_list=[polygon1, polygon2,polygon3,
                       polygon4, polygon5,polygon6,
                       polygon7, polygon8,polygon9,
                       polygon10, polygon11],
        goal_bias=0,
        step_size=3.,
        proximity_radius=3.,
        balance_ratio=0.1,
        num_iter=2000,
    )
    step = 0
    while True:
        G1.iterate()
        step +=1
        if len(G1.Path) > 0:
            print("num of steps",step)
            break 
    
    print(time.time()-t0)
    G1.test_print(ax)
  #  print("distance:",G1.distances[G1.goal_node.idx])
    plt.show()
