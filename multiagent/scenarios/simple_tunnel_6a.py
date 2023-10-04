import numpy as np
from multiagent.core import World, Agent, Landmark,Wall
from multiagent.scenario import BaseScenario
#随机生成等量目标点


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 6
        num_obstacle=2
        num_landmarks = 6

        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.accel = 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.3
            agent_id=i
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks+num_obstacle)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'target %d' % i if i<num_landmarks else 'obstacle %d' %(i-num_landmarks)
            landmark.collide = False if i < num_landmarks else True
            landmark.movable = False
            landmark.size= 0.05 if i < num_landmarks else 0.8





        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25]) if landmark.collide == True else np.array([0.85,0.85,0.85])
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        obstacles=self.obstacle(world)
        obstacles[0].state.p_pos=np.array([-1,0])
        obstacles[1].state.p_pos = np.array([1, 0])

        positions=np.linspace(-0.9,0.9,6)

        targets=self.targets(world)
        agents = world.agents
        for i in range(len(targets)):
            targets[i].state.p_pos=np.array([positions[i],0.9])
            agents[i].state.p_pos=np.array([positions[i],-0.9])


        for i, landmark in enumerate(world.landmarks):

            #landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p) if not landmark.collide else np.random.uniform(-0.5,0.5,world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = self.reward(agent,world)
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for i,l in enumerate(self.targets(world)):

            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)

            if min(dists) < 0.05:
                occupied_landmarks += 1

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
                for o in self.obstacle(world):
                    if self.is_collision(a,o):
                        rew -= 1
                        collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        shape=True
        id=agent.id
        if shape:
            target=self.targets(world)[id]

            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
            rew-=dist

            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
                for o in self.obstacle(world):
                    if self.is_collision(agent,o):
                        rew -= 1

            def bound(x):
                if x < 0.95:
                    return 0
                if x < 1.0:
                    return (x - 0.95) * 10
                return min(np.exp(2 * x - 2), 10)

            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= bound(x)
        else:
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
                for o in self.obstacle(world):
                    if self.is_collision(agent,o):
                        rew -= 1
                for t in self.targets(world):
                    if self.is_collision(agent,t):
                        rew+=10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)


    def part_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            dist = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
            if not entity.boundary and (agent.view_radius >= 0) and dist <= agent.view_radius:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(np.array([0., 0.]))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if agent.view_radius >= 0 and dist <= agent.view_radius:
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append(np.array([0., 0.]))
                if not other.adversary:
                    other_vel.append(np.array([0., 0.]))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)


    def obstacle(self,world):
        return [obstacle for obstacle in world.landmarks if obstacle.collide]

    def targets(self,world):
        return [target for target in world.landmarks if not target.collide]
