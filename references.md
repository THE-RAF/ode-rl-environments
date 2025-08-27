RL loop:

for episode in range(n_episodes):
        observation = env.reset()
        rewards_sum = 0

        done = False
        while not done:
            action = agent.take_action(observation)
            observation, reward, done = env.step(action)
            observation = observation
            print(observation)

            rewards_sum += reward

        rewards.append(rewards_sum)
    env.close()

    if render:
        env.render()

# this means that our ODE environments should have this same structure of observation, reward, done and step(action)


Model of a theoretical 2x2 ODE:

class Theoretical2x2:
    def __init__(self, parameters):
        self.parameters = {
            'x0': 1,
            'x1': 1,
            'u0': 1,
            'u1': 1,
            'y0': 1,
            'y1': 1,
        }

        self.initial_conditions = np.array([self.parameters['x0'], self.parameters['x1']])

        self.A = [[0, 1],
                  [-1, 0]]

        self.B = [[1, 0],
                  [0, 1]]

        self.C = [[1, 0],
                  [0, 1]]

        self.D = [[0, 0],
                  [0, 0]]

        self.A = np.array(self.A)
        self.B = np.array(self.B)
        self.C = np.array(self.C)
        self.D = np.array(self.D)

    def step(self, X, t):
        self.parameters['x0'] = X[0]
        self.parameters['x1'] = X[1]

        U = np.array([self.parameters['u0'], self.parameters['u1']])

        Y = self.C @ X + self.D @ U

        self.parameters['y0'] = Y[0]
        self.parameters['y1'] = Y[1]

        dXdt = self.A @ X + self.B @ U

        return dXdt

Model of a heated tank:

class HeatedTank:
    def __init__(self, parameters):
        self.parameters = parameters
        self.initial_conditions = np.array([self.parameters['Tv'], self.parameters['Tj']])

    def step(self, T, t):
        F = self.parameters['F']
        Fj = self.parameters['Fj']
        T0 = self.parameters['T0']
        Tj0 = self.parameters['Tj0']
        V = self.parameters['V']
        Vj = self.parameters['Vj']
        rho = self.parameters['rho']
        rhoj = self.parameters['rhoj']
        cp = self.parameters['cp']
        cpj = self.parameters['cpj']
        U = self.parameters['U']
        A = self.parameters['A']

        Tv = T[0]
        Tj = T[1]
        self.parameters['Tv'] = Tv
        self.parameters['Tj'] = Tj

        dTdt = np.zeros(shape=2)

        dTdt[0] = (F * (T0 - Tv))/V + (U * A * (Tj - Tv))/(V * rho * cp)
        dTdt[1] = (Fj * (Tj0 - Tj))/Vj - (U * A * (Tj - Tv))/(Vj * rhoj * cpj)

        return dTdt

# this is an example of a specific model, note that all the models follow the same structure, so we can easily swap them and pass them through a, ODE integrator