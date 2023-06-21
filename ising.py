import numpy as np
import math
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output, DisplayHandle

import multiprocess


#################################################################
###                      Helper Functions                     ###
#################################################################

def rand_field(cells, magnitude=1, domain=None):
    if domain is None:
        values = magnitude*np.random.random(len(cells))
    else:
        values = np.random.choice(domain, size=len(cells))
    return dict(zip(cells, values))

def rand_id(length=5):
    return ''.join([str(i) for i in np.random.choice(range(10),length)])

def rand_screen():
    return DisplayHandle(display_id=rand_id())

def const_field(cells, value=0):
    return dict(zip(cells, [value]*len(cells)))

def selected(upper_limit):
    assert 0 <= upper_limit <= 1, 'Probability threshold out of range [0,1].'
    return np.random.random() < upper_limit





################################### Main Class ###################################

class ising:
    #################################################################
    ###                   Model Initialization                    ###
    #################################################################

    def __init__(self, dimensions, spin_cell=0, coupling=None, mag_field=None, spins=None) -> None:
        self.DIM = np.array(dimensions)
        self.D = len(self.DIM)
        self.P = spin_cell

        self.cell0 = list(itertools.product(*[range(i) for i in self.DIM]))
        self.cellP = list(itertools.product(self.cell0, range(math.comb(self.D, self.P))))
        self.cellP1 = list(itertools.product(self.cell0, range(math.comb(self.D, self.P+1))))

        if coupling is None:
            self.coupling = const_field(self.cellP1, value=0)
        else:
            self.coupling = coupling
        
        if mag_field is None:
            self.mag_field = const_field(self.cellP, value=0)
        else:
            self.mag_field = mag_field
        
        if spins is None:
            self.spins = rand_field(self.cellP, domain=[1,-1])
        else:
            self.spins = spins

        self.P1dirs = list(itertools.combinations(range(self.D), self.P+1))
        self.Pdirs = list(itertools.combinations(range(self.D), self.P))
        self.dirsP_to_ord = dict([(x, i) for i, x in enumerate(self.Pdirs)])
        self.ord_to_dirsP1 = dict(enumerate(self.P1dirs))

        def bdry(cell, ord_to_dirs, dirs_to_ord):
            v = list(cell[0])
            dirs = list(ord_to_dirs[cell[1]])
    
            bd_list = []
            for i, d in enumerate(dirs):
                wall = dirs_to_ord[tuple(dirs[:i] + dirs[i+1:])]
                bd_list.append((tuple(v), wall))
                v_new = v.copy()
                v_new[d] = (v_new[d]+1)%self.DIM[d]
                bd_list.append((tuple(v_new), wall))
    
            return bd_list

        self.P1_to_P = {}
        self.P_to_P = defaultdict(set)
        self.P_to_P1 = defaultdict(set)
        for cell in self.cellP1:
            walls = bdry(cell, self.ord_to_dirsP1, self.dirsP_to_ord)
            self.P1_to_P[cell] = walls
            for wall in walls:
                self.P_to_P[wall] = self.P_to_P[wall].union(set(walls))
                self.P_to_P[wall].discard(wall)
                self.P_to_P1[wall].add(cell)



    #################################################################
    ###                    Action Functionals                     ###
    #################################################################
    
    def interaction(self, i_site, spin=None):
        if spin is None: spin = self.spins

        walls = self.P1_to_P[i_site]
        s_at_walls = np.array([spin[wall] for wall in walls])
        return np.prod(s_at_walls)
    

    def S(self, spin=None):
        if spin is None: spin = self.spins

        action = 0
    
        for i_site in self.coupling:
            action += self.coupling[i_site] * self.interaction(i_site, spin=spin)
    
        for site in self.mag_field:
            action += spin[site] * self.mag_field[site]
    
        return -action
    

    
    #################################################################
    ###                       Monte Carlo                         ###
    #################################################################
    
    def MC_move(self, num_iterate=1, screen=None) -> None:
        if screen is not None and num_iterate>10:
            print('', end='\r')
            screen.display('...')
            screen_counter = 0
    
        turn = 0
        while turn < num_iterate:
            turn += 1
        
            if screen is not None and num_iterate>10:
                if screen_counter%10 == 0:
                    message = 'MC turn: {}/{}'.format(round(turn), num_iterate)
                    screen.update(message)
                screen_counter += 1
        
            seed = self.cellP[np.random.choice(len(self.cellP))]

            stack = [seed]
            cluster = set(stack)

            cost_mag = 2 * self.mag_field[seed] * self.spins[seed]

            while stack:
                site = stack.pop()
                for i_site in self.P_to_P1[site]:
                    boltzman = min(1, math.exp(-2*self.coupling[i_site]))
                    if self.interaction(i_site) < 0 or selected(boltzman): continue

                    for wall in self.P1_to_P[i_site]:
                        if wall not in cluster:
                            cluster.add(wall)
                            stack.append(wall)
                            cost_mag += 2 * self.mag_field[wall] * self.spins[wall]

            if cost_mag <= 0 or selected(math.exp(-cost_mag)):
                for site in cluster:
                    self.spins[site] *= -1
                turn += len(cluster)
    
    def equilibrate(self, num_iterate=None, track_energy = False, screen=None):
        if num_iterate is None: num_iterate = 2*len(self.cellP)

        diagnostic_screen = rand_screen()
        if track_energy: 
            screen = None
            print('', end='\r')
            diagnostic_screen.display('...')
            energies = np.array([])

        if screen is not None:
            print('', end='\r')
            screen.display('...')

        for turn in range(num_iterate):
            self.MC_move(screen=screen)

            if screen is not None and turn%50==0:
                message = 'MC turn: {}/{}'.format(turn+1, num_iterate)
                screen.update(message)

            if track_energy:
                energies = np.append(energies, self.S())
                message = 'MC_move: {}/{}. Energy: {:.2f}'.format(turn, num_iterate, self.S())
                diagnostic_screen.update(message)
        
        if track_energy:
            message = 'std as % of mean energy: {:.3f}'.format(100*energies.std()/abs(energies.mean()))
            diagnostic_screen.update(message)




    #################################################################
    ###                    Expectation Values                     ###
    #################################################################

    def expectation_single_process(self, operator, sample_size, duality_frame='primal', screen=None):
        E_val = 0
    
        screen_counter = 0
        if screen is not None:
            print('', end='\r')
            screen.display('...')
    
        for turn in range(sample_size):
            if screen is not None:
                if screen_counter%1 == 0:
                    message = 'E sample (J = {:.3f}): {}/{}. running E: {:.3f}'.format(self.coupling[self.cellP1[0]], turn+1, sample_size, E_val/turn if turn>0 else 0)
                    screen.update(message)
                screen_counter += 1

            #self.spins = rand_field(self.cellP, domain=[1,-1])
            #self.equilibrate()
            self.MC_move(num_iterate=2*len(self.cellP))

            o_val = 1
            for site in operator:
                if duality_frame == 'primal':
                    per_site = self.spins[site]
                elif duality_frame == 'dual':
                    per_site = math.exp(- 2 * self.coupling[site] * self.interaction(site))
                o_val *= per_site
            E_val += o_val

        return E_val/sample_size
    
    def expectation(self, operator, sample_size, n_processors=1, duality_frame='primal', quiet=True):
        n_processors = n_processors
        pool = multiprocess.Pool(processes=n_processors)

        screens = [rand_screen() for _ in range(n_processors)]

        expectation_partial = lambda screen: self.expectation_single_process(operator, sample_size, duality_frame=duality_frame, screen=screen if not quiet else None)

        return sum(pool.map(expectation_partial, screens))/n_processors
    



    #################################################################
    ###                         Duality                           ###
    #################################################################

    def dual(self):
        dual_coupling = {}
        for site in self.cellP:
            dual_coupling[site] = math.atanh(math.exp(-2*self.mag_field[site]))

        dual_mag_field = {}
        for i_site in self.cellP1:
            dual_mag_field[i_site] = math.atanh(math.exp(-2*self.coupling[i_site]))

        return ising(self.DIM, spin_cell=self.D-self.P-1, coupling=dual_coupling, mag_field=dual_mag_field)
    

    

    #################################################################
    ###        Methods for Information Display/Diagnostics        ###
    #################################################################
    
    def show_spins(self):
        if not (self.D==2 and self.P==0):
            print('This method is available only for D=2 and P=0. The current model has D={} and P={}.'.format(self.D, self.P))
            return

        a = np.full(self.DIM, 0)
        for site in self.spins:
            (x, y), _ = site
            a[x][y] = self.spins[site]
        ups = 100*(a.sum()+len(self.cellP))/(2*len(self.cellP))
        downs = 100-ups
        print('% of (+, -) spins: ({:.2f}, {:.2f})'.format(ups, downs))
        sns.heatmap(a, cmap='binary', vmin=-1, vmax=1)
        plt.show()
    
    def info(self, show_spins=False) -> None:
        print('Lattice dimension:', self.D)
        print('Lattice size:', self.DIM)
        print('Spins are placed on:', str(self.P) + '-cells')
        print('Interactions are placed on:', str(self.P+1) + '-cells')
        print('No. of ' + str(self.P) + '-cells in lattice:', np.prod(self.DIM) * math.comb(self.D, self.P))
        print('No. of ' + str(self.P+1) + '-cells in lattice:', np.prod(self.DIM) * math.comb(self.D, self.P+1))

        if show_spins: self.show_spins()