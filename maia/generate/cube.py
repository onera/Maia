import time
import fnmatch
from math import *

import numpy as np

import Converter.PyTree   as C
import Converter.Internal as I
import Transform.PyTree   as T
import Generator.PyTree   as G
import Connector.PyTree   as X
import Post.PyTree        as P
import Converter.elsAProfile as elsAProfile

import etc.transform.__future__ as trf


# ---------------------------------------------------------------------
getFamilys = lambda t,f:[n for n in I.getNodesByType(t, 'Family_t') if fnmatch.fnmatch(I.getName(n), f)]
getFamily  = lambda t,f:getFamilys(t,f)[0]


# --------------------------------------------------
def newFamilyBCDataSet(name='FamilyBCDataSet', parent=None):
    """Create a new FamilyBCDataSet node."""
    if parent is None:
        node = I.createNode(name, 'FamilyBCDataSet_t')
    else: node = I.createUniqueChild(parent, name, 'FamilyBCDataSet_t')
    return node

# ---------------------------------------------------------------------
def search_zone_wrt_x(t, conditions):
    zones = []
    for zone in I.getZones(t):
        X = I.getNodeFromNameAndType(zone, 'CoordinateX', 'DataArray_t')
        x = I.getValue(X)
        zone_type_node = I.getNodeFromType(zone, 'ZoneType_t')
        zone_type = I.getValue(zone_type_node)
        if zone_type == "Structured":
            x = x.flatten()
        if all(all(c(x)) for c in conditions):
            zones.append(zone)
    return zones

def search_bc(bcs, condition):
    zones = []
    for zone in I.getZones(bcs):
        # print("Search BC in zone {}".format(I.getName(zone)))
        X = I.getNodeFromNameAndType(zone, 'CoordinateX', 'DataArray_t')
        Y = I.getNodeFromNameAndType(zone, 'CoordinateY', 'DataArray_t')
        Z = I.getNodeFromNameAndType(zone, 'CoordinateZ', 'DataArray_t')
        x = I.getValue(X)
        y = I.getValue(Y)
        z = I.getValue(Z)
        if all(condition(x, y, z)):
            # print("found '{}' BC {}".format(name, I.getName(z)))
            zones.append(zone)
    return zones

def add_bc(bc, name, BCs, BCName, BCType):
    BCs    += [bc]
    BCName += ["BC{}".format(name)]
    BCType += ["FamilySpecified:{}".format(name)]

def create_periodic_join(t, translation):
    t = X.connectMatchPeriodic(t, translation=translation, tol=1.e-4)
    return t

def create_tree(t):
    tree = I.newCGNSTree()
    base = I.newCGNSBase('cgns_base', 3, 3, parent=tree)
    for zone in I.getZones(t):
        I.addChild(base, zone)
    return tree, base

def fix_gc_value(t):
    # Correction for GridConnectivity value
    base2zone = dict([(I.getName(b),[I.getName(z) for z in I.getZones(b)]) for b in I.getBases(t)])
    for b in I.getBases(t):
        for gc in I.getNodesFromType(b, "GridConnectivity_t")+I.getNodesFromType(b, "GridConnectivity1to1_t"):
          for base, zones in base2zone.items():
            if I.getValue(gc) in zones and base != I.getName(b):
                I.setValue(gc, f"{base}/{I.getValue(gc)}")
    return t

def create_bc(t, p0, h0, ps):
    tree, base = create_tree(t)

    allbcs = []
    f = I.newFamily(name='Inlet', parent=base)
    inlet = trf.BCInj1(tree, f, stagnation_pressure=p0,
                                stagnation_enthalpy=h0,
                                tv=(1., 0., 0.))
    allbcs.append(inlet)
    f = I.newFamily(name='Wall', parent=base)
    wall = trf.BCWallSlip(tree, f)
    allbcs.append(wall)

    # With Family_t
    foutlet = I.newFamily(name='Outlet', parent=base)
    outlet = trf.BCOutPres(tree, foutlet)
    outlet.pressure = ps
    allbcs.append(outlet)

    # With BC_t
    bc = None
    for n in I.getNodesByType(base, 'BC_t'):
        fn = I.getNodeByType(n, "FamilyName_t")
        if fnmatch.fnmatch(I.getValue(fn), 'Outlet'):
            bc = n
            break
    I.printTree(bc)
    outlet = trf.BCOutPres(tree, bc)
    bcsize = outlet.getBCSize()
    pressure = np.ndarray(bcsize, dtype=np.float64, order='F')
    pressure.fill(ps)
    outlet.pressure = pressure
    allbcs.append(outlet)

    for bc in allbcs:
        bc.create()
    I._rmNode(foutlet, I.getNodeFromName(foutlet, ".Solver#BC"))
    I.printTree(tree)

    # # With FamilyBCDataSet
    # fbc_node   = I.getNodeByType(foutlet, "FamilyBC_t")
    # fbcds_node = newFamilyBCDataSet(parent=fbc_node)
    # bcd_node   = I.newBCData(parent=fbcds_node, name="Neumann")
    # pressure = np.ndarray(1, dtype=np.float64, order='F')
    # pressure.fill(ps)
    # I.newDataArray(parent=bcd_node, name="Pressure", value=pressure)
    # I.printTree(tree)
    return tree

def create_bc1(t, b, p0, h0):
    allbcs = []
    f = I.newFamily(name='Inlet', parent=b)
    inlet = trf.BCInj1(t, f, stagnation_pressure=p0,
                                stagnation_enthalpy=h0,
                                tv=(1., 0., 0.))
    allbcs.append(inlet)
    f = I.newFamily(name='Wall', parent=b)
    wall = trf.BCWallSlip(t, f)
    allbcs.append(wall)

    for bc in allbcs:
        bc.create()
    # I.printTree(tree)

    return t

def create_bc2(t, b, ps):
    allbcs = []
    f = I.newFamily(name='Wall', parent=b)
    wall = trf.BCWallSlip(t, f)
    allbcs.append(wall)
    f = I.newFamily(name='Outlet', parent=b)
    outlet = trf.BCOutPres(t, f, pressure=ps)
    allbcs.append(outlet)

    for bc in allbcs:
        bc.create()
    # I.printTree(tree)

    return t

def create_motion(t, motionPerRow, conditions):
    # Add motion
    for b in I.getBases(t):
        zones = search_zone_wrt_x(t, conditions)
        for name, omg in list(motionPerRow.items()):
            for zone in zones:
              fn = I.newFamilyName(name='FamilyName', value=name, parent=zone)

            # Get 'Row1', 'Row2' family
            f = I.newFamily(name=name, parent=b)
            # Add a elsA/CGNS Trigger
            mtn = trf.FamilyMotion(f, motion='mobile',
                                      omega=omg,
                                      axis_pnt=(0.,0.,0.),
                                      axis_vct=(1.,0.,0.))
            mtn.create()
    return t


# ---------------------------------------------------------------------
class State(object):

    def __init__(self):
        self.eps = 1.e-12
        self.x0, self.y0, self.z0 = (0.,)*3
        self.dx, self.dy, self.dz = (1.,)*3
        self.nx, self.ny, self.nz = (4,)*3
        self.x1 = self.x0 + (self.nx-1)*self.dx
        self.y1 = self.y0 + (self.ny-1)*self.dy
        self.z1 = self.z0 + (self.nz-1)*self.dz

        self.Rgaz     = 287.04
        self.Gamma    = 1.4
        self.cv       = self.Rgaz/(self.Gamma-1.)

        self.Tio      = 288.14
        self.Pio      = 101325.
        self.roio     = self.Pio / ((self.Gamma-1.)*self.cv*self.Tio)
        self.aio      = sqrt(self.Gamma*self.Pio/self.roio)

        self.Tparoi   = 288.5
        self.Lref     = 1.e-3
        self.csuth    = 110.4
        self.Tsuth    = 273.
        self.musuth   = 1.711E-5

        self.cv_a     = 1./((self.Gamma-1.)*self.Gamma)
        self.Tio_a    = 1.
        self.Pio_a    = 1. / self.Gamma
        self.Hio_a    = 1. / (self.Gamma-1.)
        self.Ps_a     = 0.9*self.Pio_a

        # Initialize
        self.Tp_a     = self.Tparoi/self.Tio
        self.csuth_a  = self.csuth / self.Tio
        self.Tsuth_a  = self.Tsuth / self.Tio
        self.musuth_a = self.musuth / (self.roio*self.aio*self.Lref)

        self.Mach     = 0.2
        self.TsTio    = (1.+0.5*(self.Gamma-1.)*self.Mach*self.Mach)**(-1.)
        self.Uext     = self.Mach*sqrt(self.Gamma*(self.Gamma-1.)*self.cv*self.TsTio*self.Tio)

        self.ro_ini   = (1.+0.5*(self.Gamma-1.)*self.Mach*self.Mach)**(1./(1.-self.Gamma))
        self.rou_ini  = (self.ro_ini*self.Uext/self.aio)*cos(0.*pi/180.)
        self.rov_ini  = (self.ro_ini*self.Uext/self.aio)*sin(0.*pi/180.)
        self.row_ini  = 0.
        self.roe_ini  = (self.cv_a*self.TsTio+0.5*self.Uext*self.Uext/(self.aio*self.aio))*self.ro_ini

        self.k1_ini   = 1.e-08
        self.k2_ini   = 1.e-08

        # Motion
        self.omega1   = 6500.*pi/30.
        # self.omega2   = 0.
        self.omega1_a = self.omega1*self.Lref/self.aio
        # self.omega2_a = self.omega2*self.Lref/self.aio

        self.motionPerRow = {'Row1' : self.omega1_a}

        self.Naube1   = 19
        # self.Naube2   = 36

        self.Nduplic1 = 1
        # self.Nduplic2 = 1

        self.Cq1      = self.Naube1*self.roio*self.aio*self.Lref*self.Lref
        # self.Cq2      = self.Naube2*self.roio*self.aio*self.Lref*self.Lref


# ---------------------------------------------------------------------
class CubeFactory(object):

    def __init__(self, state):
        self.state = state
        self.xyz   = [(state.x0, state.y0, state.z0),
                      (state.dx, state.dy, state.dz),
                      (state.nx, state.ny, state.nz)]
        self.xlimits = [lambda x : x > state.x0-state.eps,
                        lambda x : x < state.x1+state.eps]
        self.xlimits2 = [lambda x : x > state.x0-state.eps,
                         lambda x : x < 2.*state.x1+state.eps]

    def add_mean_flow_solution(self, t):
        n_cell = (self.state.nx-1)*(self.state.ny-1)*(self.state.nz-1)
        # Add Flow Solution for mean flow
        fs_node = I.newFlowSolution(name='FlowSolution#Init', gridLocation='CellCenter')
        density = np.zeros(n_cell, dtype=np.float64, order='F')
        density.fill(self.state.ro_ini)
        density_node = I.newDataArray("Density", value=density, parent=fs_node)
        for i,j in zip(("X","Y","Z"), (self.state.rou_ini, self.state.rov_ini, self.state.row_ini)):
          momentum = np.zeros(n_cell, dtype=np.float64, order='F')
          momentum.fill(j)
          momentum_node = I.newDataArray(f"Momentum{i}", value=momentum, parent=fs_node)
        energy_stagnation_energy = np.zeros(n_cell, dtype=np.float64, order='F')
        energy_stagnation_energy.fill(self.state.roe_ini)
        energy_stagnation_energy_node = I.newDataArray("EnergyStagnationDensity",
                                                       value=energy_stagnation_energy,
                                                       parent=fs_node)
        for zone in I.getZones(t):
            I.addChild(zone, fs_node)
        return t

    def add_mean_reference_state(self, t):
        # Add reference state
        for base in I.getBases(t):
            r = I.newReferenceState(parent=base, name="ReferenceState")
            rfs = trf.ReferenceState(r)
            rfs.state = {"Density"                       : self.state.ro_ini,
                         "MomentumX"                     : self.state.rou_ini,
                         "MomentumY"                     : self.state.rov_ini,
                         "MomentumZ"                     : self.state.row_ini,
                         "EnergyStagnationDensity"       : self.state.roe_ini}
            rfs.create()
        return t

    # ==========================
    # Structured cube generation
    # ==========================
    def cube_structured(self):
        t = G.cart(*self.xyz)
        t, _ = create_tree(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    def cube_structured_alljoin(self):
        t = G.cart(*self.xyz)
        t = create_periodic_join(t, (self.state.x1,           0.,           0.))
        t = create_periodic_join(t, (           0.,self.state.y1,           0.))
        t = create_periodic_join(t, (           0.,           0.,self.state.z1))
        t, _ = create_tree(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    def cube_structured_allbnd(self):
        t = G.cart(*self.xyz)
        t = C.addBC2Zone(t, 'inlet',  'FamilySpecified:Inlet',  'imin')
        t = C.addBC2Zone(t, 'outlet', 'FamilySpecified:Outlet', 'imax')
        t = C.addBC2Zone(t, 'wall',   'FamilySpecified:Wall',   'jmin')
        t = C.addBC2Zone(t, 'wall',   'FamilySpecified:Wall',   'jmax')
        t = C.addBC2Zone(t, 'wall',   'FamilySpecified:Wall',   'kmin')
        t = C.addBC2Zone(t, 'wall',   'FamilySpecified:Wall',   'kmax')
        t = create_bc(t, self.state.Pio_a, self.state.Hio_a, self.state.Ps_a)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    def cube_structured_join_bnd(self):
        t = G.cart(*self.xyz)
        t = C.addBC2Zone(t, 'inlet',  'FamilySpecified:Inlet',  'imin')
        t = C.addBC2Zone(t, 'outlet', 'FamilySpecified:Outlet', 'imax')
        t = C.addBC2Zone(t, 'wall',   'FamilySpecified:Wall',   'jmin')
        t = C.addBC2Zone(t, 'wall',   'FamilySpecified:Wall',   'jmax')
        t = create_periodic_join(t, (0.,0.,self.state.z1))
        t = create_bc(t, self.state.Pio_a, self.state.Hio_a, self.state.Ps_a)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    # Structured/Structured NGon cube generation for multi-bases
    # ----------------------------------------------------------
    def _cube_structured_multi_bases(self):
        # Create unstructured zone with CGNS base 1
        t1 = G.cart(*self.xyz)
        # Create unstructured zone with CGNS base 2
        t2 = G.cart(*self.xyz)
        T._translate(t2, (self.state.x1,0.,0.))
        return t1, t2

    def cube_structured_multi_bases(self):
        # Create structured zone
        t1, t2 = self._cube_structured_multi_bases()
        # Create CGNS Tree
        t = I.newCGNSTree()
        b1 = I.newCGNSBase('cgns_base_1', 3, 3, parent=t)
        for zone in I.getZones(t1):
            I.addChild(b1, zone)
        b2 = I.newCGNSBase('cgns_base_2', 3, 3, parent=t)
        for zone in I.getZones(t2):
            I.addChild(b2, zone)
        # Create join
        t = X.connectMatch(t, tol=1.e-6)
        # Correction for GridConnectivity value
        t = fix_gc_value(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t

    def cube_structured_join_bnd_multi_bases(self):
        # Create structured zone
        t1, t2 = self._cube_structured_multi_bases()
        # Create BC(s)
        t1 = C.addBC2Zone(t1, 'inlet',  'FamilySpecified:Inlet',  'imin')
        t1 = C.addBC2Zone(t1, 'wall',   'FamilySpecified:Wall',   'jmin')
        t1 = C.addBC2Zone(t1, 'wall',   'FamilySpecified:Wall',   'jmax')
        t2 = C.addBC2Zone(t2, 'outlet', 'FamilySpecified:Outlet', 'imax')
        t2 = C.addBC2Zone(t2, 'wall',   'FamilySpecified:Wall',   'jmin')
        t2 = C.addBC2Zone(t2, 'wall',   'FamilySpecified:Wall',   'jmax')
        # Create CGNS Tree
        t = I.newCGNSTree()
        b1 = I.newCGNSBase('cgns_base_1', 3, 3, parent=t)
        for zone in I.getZones(t1):
            I.addChild(b1, zone)
        b2 = I.newCGNSBase('cgns_base_2', 3, 3, parent=t)
        for zone in I.getZones(t2):
            I.addChild(b2, zone)
        t = create_bc1(t, b1, self.state.Pio_a, self.state.Hio_a)
        t = create_bc2(t, b2, self.state.Ps_a)
        # Create join
        t = X.connectMatch(t, tol=1.e-6)
        t = create_periodic_join(t, (0.,0.,self.state.z1))
        # Correction for GridConnectivity value
        t = fix_gc_value(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t

    # =================================
    # Unstructured NGon cube generation
    # =================================
    def cube_unstructured(self):
        t = G.cartNGon(*self.xyz)
        I._fixNGon(t)
        elsAProfile._createElsaHybrid(t)
        t, _ = create_tree(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    def cube_unstructured_alljoin(self):
        t = G.cartNGon(*self.xyz)
        I._fixNGon(t)
        elsAProfile._createElsaHybrid(t)
        # Create join(s)
        t = create_periodic_join(t, (self.state.x1,           0.,           0.))
        t = create_periodic_join(t, (           0.,self.state.y1,           0.))
        t = create_periodic_join(t, (           0.,           0.,self.state.z1))
        t, _ = create_tree(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    def cube_unstructured_allbnd(self):
        t = G.cartNGon(*self.xyz)
        I._fixNGon(t)
        elsAProfile._createElsaHybrid(t)
        # Create BC(s)
        ef  = P.exteriorFaces(t)
        bcs = T.splitSharpEdges(ef, 20.)
        inlet  = search_bc(bcs, lambda x,y,z : x < self.state.x0+self.state.eps)
        outlet = search_bc(bcs, lambda x,y,z : x > self.state.x1-self.state.eps)
        wally0 = search_bc(bcs, lambda x,y,z : y < self.state.y0+self.state.eps)
        wally1 = search_bc(bcs, lambda x,y,z : y > self.state.y1-self.state.eps)
        wallz0 = search_bc(bcs, lambda x,y,z : z < self.state.z0+self.state.eps)
        wallz1 = search_bc(bcs, lambda x,y,z : z > self.state.z1-self.state.eps)
        BCs = []; BCName = []; BCType = []
        add_bc(inlet,  "Inlet",  BCs, BCName, BCType)
        add_bc(outlet, "Outlet", BCs, BCName, BCType)
        add_bc(wally0, "Wall",   BCs, BCName, BCType)
        add_bc(wally1, "Wall",   BCs, BCName, BCType)
        add_bc(wallz0, "Wall",   BCs, BCName, BCType)
        add_bc(wallz1, "Wall",   BCs, BCName, BCType)
        C._recoverBCs(t, (BCs, BCName, BCType))
        t = create_bc(t, self.state.Pio_a, self.state.Hio_a, self.state.Ps_a)
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    def cube_unstructured_join_bnd(self):
        t = G.cartNGon(*self.xyz)
        I._fixNGon(t)
        elsAProfile._createElsaHybrid(t)
        # Create BC(s)
        ef  = P.exteriorFaces(t)
        bcs = T.splitSharpEdges(ef, 20.)
        inlet  = search_bc(bcs, lambda x,y,z : x < self.state.x0+self.state.eps)
        outlet = search_bc(bcs, lambda x,y,z : x > self.state.x1-self.state.eps)
        wally0 = search_bc(bcs, lambda x,y,z : y < self.state.y0+self.state.eps)
        wally1 = search_bc(bcs, lambda x,y,z : y > self.state.y1-self.state.eps)
        BCs = []; BCName = []; BCType = []
        add_bc(inlet,  "Inlet",  BCs, BCName, BCType)
        add_bc(outlet, "Outlet", BCs, BCName, BCType)
        add_bc(wally0, "Wall",   BCs, BCName, BCType)
        add_bc(wally1, "Wall",   BCs, BCName, BCType)
        C._recoverBCs(t, (BCs, BCName, BCType))
        t = create_bc(t, self.state.Pio_a, self.state.Hio_a, self.state.Ps_a)
        # Create join(s)
        t = create_periodic_join(t, (0.,0.,self.state.z1))
        t = create_motion(t, self.state.motionPerRow, self.xlimits)
        return t

    # Unstructured/Unstructured NGon cube generation for multi-bases
    # --------------------------------------------------------------
    def _cube_unstructured_multi_bases(self):
        # Create unstructured zone with CGNS base 1
        t1 = G.cartNGon(*self.xyz)
        # Create unstructured zone with CGNS base 2
        t2 = G.cartNGon(*self.xyz)
        T._translate(t2, (self.state.x1,0.,0.))
        return t1, t2

    def cube_unstructured_multi_bases(self):
        # Create structured zone
        t1, t2 = self._cube_unstructured_multi_bases()
        # Create CGNS Tree
        t = I.newCGNSTree()
        b1 = I.newCGNSBase('cgns_base_1', 3, 3, parent=t)
        for zone in I.getZones(t1):
            I.addChild(b1, zone)
        b2 = I.newCGNSBase('cgns_base_2', 3, 3, parent=t)
        for zone in I.getZones(t2):
            I.addChild(b2, zone)
        I._fixNGon(t)
        elsAProfile._createElsaHybrid(t)
        # Create join
        t = X.connectMatch(t, tol=1.e-6)
        # Correction for GridConnectivity value
        t = fix_gc_value(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t

    def cube_unstructured_join_bnd_multi_bases(self):
        # Create structured zone
        t1, t2 = self._cube_unstructured_multi_bases()
        # Create BC(s)
        ef1  = P.exteriorFaces(t1)
        bcs1 = T.splitSharpEdges(ef1, 20.)
        # C.convertPyTree2File(bcs1, "bcs1.adf", format="bin_adf")
        inlet  = search_bc(bcs1, lambda x,y,z : x < self.state.x0+self.state.eps)
        wally0 = search_bc(bcs1, lambda x,y,z : y < self.state.y0+self.state.eps)
        wally1 = search_bc(bcs1, lambda x,y,z : y > self.state.y1-self.state.eps)
        BCs = []; BCName = []; BCType = []
        add_bc(inlet,  "Inlet",  BCs, BCName, BCType)
        add_bc(wally0, "Wall",   BCs, BCName, BCType)
        add_bc(wally1, "Wall",   BCs, BCName, BCType)
        C._recoverBCs(t1, (BCs, BCName, BCType))
        ef2  = P.exteriorFaces(t2)
        bcs2 = T.splitSharpEdges(ef2, 20.)
        # C.convertPyTree2File(bcs2, "bcs2.adf", format="bin_adf")
        outlet = search_bc(bcs2, lambda x,y,z : x > 2.*self.state.x1-self.state.eps)
        wally0 = search_bc(bcs2, lambda x,y,z : y < self.state.y0+self.state.eps)
        wally1 = search_bc(bcs2, lambda x,y,z : y > self.state.y1-self.state.eps)
        BCs = []; BCName = []; BCType = []
        add_bc(outlet, "Outlet", BCs, BCName, BCType)
        add_bc(wally0, "Wall",   BCs, BCName, BCType)
        add_bc(wally1, "Wall",   BCs, BCName, BCType)
        C._recoverBCs(t2, (BCs, BCName, BCType))
        # Create CGNS Tree
        t = I.newCGNSTree()
        b1 = I.newCGNSBase('cgns_base_1', 3, 3, parent=t)
        for zone in I.getZones(t1):
            I.addChild(b1, zone)
        b2 = I.newCGNSBase('cgns_base_2', 3, 3, parent=t)
        for zone in I.getZones(t2):
            I.addChild(b2, zone)
        I._fixNGon(t)
        elsAProfile._createElsaHybrid(t)
        t = create_bc1(t, b1, self.state.Pio_a, self.state.Hio_a)
        t = create_bc2(t, b2, self.state.Ps_a)
        # Create join
        t = X.connectMatch(t, tol=1.e-6)
        t = create_periodic_join(t, (0.,0.,self.state.z1))
        # Correction for GridConnectivity value
        t = fix_gc_value(t)
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        # I.printTree(t)
        return t

    # ============================================
    # Structured/Unstructured NGon cube generation
    # ============================================
    def _cube_hybrid(self):
        # Create structured zone
        ts = G.cart(*self.xyz)
        # Create unstructured zone
        tu = G.cartNGon(*self.xyz)
        I._fixNGon(tu)
        elsAProfile._createElsaHybrid(tu)
        T._translate(tu, (self.state.x1,0.,0.))
        # Create CGNS Tree
        t, b = create_tree(ts)
        for zone in I.getZones(tu):
            I.addChild(b, zone)
        # Create hybrid join
        t = X.connectMatch(t, tol=1.e-6)
        return t, (ts, tu)

    def cube_hybrid(self):
        # Create structured zone
        t, _ = self._cube_hybrid()
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t

    def cube_hybrid_alljoin(self):
        t, _ = self._cube_hybrid()
        # Create join(s)
        t = create_periodic_join(t, (2.*self.state.x1,           0.,           0.))
        t = create_periodic_join(t, (              0.,self.state.y1,           0.))
        t = create_periodic_join(t, (              0.,           0.,self.state.z1))
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t

    def cube_hybrid_allbnd(self):
        # Create structured zone
        ts = G.cart(*self.xyz)
        C._addBC2Zone(ts, 'inlet',  'FamilySpecified:Inlet',  'imin')
        C._addBC2Zone(ts, 'wall',   'FamilySpecified:Wall',   'jmin')
        C._addBC2Zone(ts, 'wall',   'FamilySpecified:Wall',   'jmax')
        C._addBC2Zone(ts, 'wall',   'FamilySpecified:Wall',   'kmin')
        C._addBC2Zone(ts, 'wall',   'FamilySpecified:Wall',   'kmax')

        # Create unstructured zone
        tu = G.cartNGon(*self.xyz)
        I._fixNGon(tu)
        elsAProfile._createElsaHybrid(tu)
        # Create BC(s)
        ef  = P.exteriorFaces(tu)
        bcs = T.splitSharpEdges(ef, 20.)
        outlet = search_bc(bcs, lambda x,y,z : x > self.state.x1-self.state.eps)
        wally0 = search_bc(bcs, lambda x,y,z : y < self.state.y0+self.state.eps)
        wally1 = search_bc(bcs, lambda x,y,z : y > self.state.y1-self.state.eps)
        wallz0 = search_bc(bcs, lambda x,y,z : z < self.state.z0+self.state.eps)
        wallz1 = search_bc(bcs, lambda x,y,z : z > self.state.z1-self.state.eps)
        BCs = []; BCName = []; BCType = []
        add_bc(outlet, "Outlet", BCs, BCName, BCType)
        add_bc(wally0, "Wall",   BCs, BCName, BCType)
        add_bc(wally1, "Wall",   BCs, BCName, BCType)
        add_bc(wallz0, "Wall",   BCs, BCName, BCType)
        add_bc(wallz1, "Wall",   BCs, BCName, BCType)
        C._recoverBCs(tu, (BCs, BCName, BCType))
        T._translate(tu, (self.state.x1,0.,0.))

        # Create CGNS Tree
        t, b = create_tree(ts)
        for zone in I.getZones(tu):
            I.addChild(b, zone)
        # Create hybrid join
        t = X.connectMatch(t, tol=1.e-6)

        t = create_bc(t, self.state.Pio_a, self.state.Hio_a, self.state.Ps_a)
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t

    def cube_hybrid_join_bnd(self):
        # Create structured zone
        ts = G.cart(*self.xyz)
        C._addBC2Zone(ts, 'inlet',  'FamilySpecified:Inlet',  'imin')
        C._addBC2Zone(ts, 'wall',   'FamilySpecified:Wall',   'jmin')
        C._addBC2Zone(ts, 'wall',   'FamilySpecified:Wall',   'jmax')
        ts = create_periodic_join(ts, (0.,0.,self.state.z1))

        # Create unstructured zone
        tu = G.cartNGon(*self.xyz)
        I._fixNGon(tu)
        elsAProfile._createElsaHybrid(tu)
        # Create BC(s)
        ef  = P.exteriorFaces(tu)
        bcs = T.splitSharpEdges(ef, 20.)
        outlet = search_bc(bcs, lambda x,y,z : x > self.state.x1-self.state.eps)
        wally0 = search_bc(bcs, lambda x,y,z : y < self.state.y0+self.state.eps)
        wally1 = search_bc(bcs, lambda x,y,z : y > self.state.y1-self.state.eps)
        BCs = []; BCName = []; BCType = []
        add_bc(outlet, "Outlet", BCs, BCName, BCType)
        add_bc(wally0, "Wall",   BCs, BCName, BCType)
        add_bc(wally1, "Wall",   BCs, BCName, BCType)
        C._recoverBCs(tu, (BCs, BCName, BCType))
        tu = create_periodic_join(tu, (0.,0.,self.state.z1))
        T._translate(tu, (self.state.x1,0.,0.))

        # Create CGNS Tree
        t, b = create_tree(ts)
        for zone in I.getZones(tu):
            I.addChild(b, zone)
        # Create hybrid join
        t = X.connectMatch(t, tol=1.e-6)

        t = create_bc(t, self.state.Pio_a, self.state.Hio_a, self.state.Ps_a)
        t = create_motion(t, self.state.motionPerRow, self.xlimits2)
        return t


if __name__ == "__main__":
    state   = State()
    factory = CubeFactory(state)
    # # ==========================
    # # Structured cube generation
    # # ==========================
    # start_time = time.time()
    # t = factory.cube_structured()
    # cube_s = (time.time() - start_time)
    # print(f"time structured cartesian mesh : {cube_s}")
    # C.convertPyTree2File(t, 'cubeS.hdf')

    # # All join
    # # --------
    # start_time = time.time()
    # t = factory.cube_structured_alljoin()
    # cube_alljoin_s = (time.time() - start_time)
    # print(f"time structured cartesian mesh with all join : {cube_alljoin_s}")
    # C.convertPyTree2File(t, 'cubeS_alljoin.hdf')

    # # All bnd
    # # -------
    # start_time = time.time()
    # t = factory.cube_structured_allbnd()
    # cube_allbnd_s = (time.time() - start_time)
    # print(f"time structured cartesian mesh with all bnd : {cube_allbnd_s}")
    # C.convertPyTree2File(t, 'cubeS_allbnd.hdf')

    # # join+bnd
    # # --------
    # start_time = time.time()
    # t = factory.cube_structured_join_bnd()
    # cube_join_bnd_s = (time.time() - start_time)
    # print(f"time structured cartesian mesh with join and bnd : {cube_join_bnd_s}")
    # C.convertPyTree2File(t, 'cubeS_join_bnd.hdf')
    # # sys.exit(1)

    # # Structured/Structured cube generation for multi-bases
    # # -----------------------------------------------------
    # t = factory.cube_structured_multi_bases()
    # # C.convertPyTree2File(t, 'cubeS_mb.adf', format='bin_adf')
    # C.convertPyTree2File(t, 'cubeS_mb.hdf')
    # # sys.exit(1)

    # # join+bnd
    # # --------
    # t = factory.cube_structured_join_bnd_multi_bases()
    # # C.convertPyTree2File(t, 'cubeS_mb_join_bnd.adf', format='bin_adf')
    # C.convertPyTree2File(t, 'cubeS_join_bnd_mb.hdf')
    # # sys.exit(1)

    # # # =================================
    # # # Unstructured NGon cube generation
    # # # =================================
    # start_time = time.time()
    # t = factory.cube_unstructured()
    # cube_u = (time.time() - start_time)
    # print(f"time unstructured cartesian mesh : {cube_u}")
    # C.convertPyTree2File(t, 'cubeU.hdf')

    # # All join
    # # --------
    # start_time = time.time()
    # t = factory.cube_unstructured_alljoin()
    # cube_alljoin_u = (time.time() - start_time)
    # print(f"time unstructured cartesian mesh with all join : {cube_alljoin_u}")
    # C.convertPyTree2File(t, 'cubeU_alljoin.hdf')

    # # All bnd
    # # -------
    # start_time = time.time()
    # t = factory.cube_unstructured_allbnd()
    # cube_allbnd_u = (time.time() - start_time)
    # print(f"time unstructured cartesian mesh with all bnd : {cube_allbnd_u}")
    # C.convertPyTree2File(t, 'cubeU_allbnd.hdf')

    # # join+bnd
    # # --------
    # start_time = time.time()
    # t0 = factory.cube_unstructured_join_bnd()
    # cube_join_bnd_u = (time.time() - start_time)
    # print(f"time unstructured cartesian mesh with join and bnd : {cube_join_bnd_u}")
    # C.convertPyTree2File(t0, 'cubeU_join_bnd.hdf')
    # t = factory.add_mean_flow_solution(t0)
    # C.convertPyTree2File(t, 'cubeU_join_bnd_fs_euler.hdf')
    # # I.printTree(t)
    # t = factory.add_mean_reference_state(t0)
    # C.convertPyTree2File(t, 'cubeU_join_bnd_fs_rf_euler.hdf')
    # # I.printTree(t)
    # t0 = factory.cube_unstructured_join_bnd()
    # t = factory.add_mean_reference_state(t0)
    # C.convertPyTree2File(t, 'cubeU_join_bnd_rf_euler.hdf')

    # # Unstructured/Unstructured NGon cube generation for multi-bases
    # # --------------------------------------------------------------
    # t = factory.cube_unstructured_multi_bases()
    # # C.convertPyTree2File(t, 'cubeU_mb.adf', format='bin_adf')
    # C.convertPyTree2File(t, 'cubeU_mb.hdf')
    # # sys.exit(1)

    # # join+bnd
    # # --------
    # t = factory.cube_unstructured_join_bnd_multi_bases()
    # # C.convertPyTree2File(t, 'cubeU_mb_join_bnd.adf', format='bin_adf')
    # C.convertPyTree2File(t, 'cubeU_mb.hdf')
    # # sys.exit(1)

    # ============================================
    # Structured/Unstructured NGon cube generation
    # ============================================
    start_time = time.time()
    t = factory.cube_hybrid()
    cube_h = (time.time() - start_time)
    print(f"time hybrid cartesian mesh : {cube_h}")
    C.convertPyTree2File(t, 'cubeH.hdf')

    # All join
    # --------
    start_time = time.time()
    t = factory.cube_hybrid_alljoin()
    cube_alljoin_h = (time.time() - start_time)
    print(f"time hybrid cartesian mesh with all join : {cube_alljoin_h}")
    C.convertPyTree2File(t, 'cubeH_alljoin.hdf')

    # All bnd
    # -------
    start_time = time.time()
    t = factory.cube_hybrid_allbnd()
    cube_allbnd_h = (time.time() - start_time)
    print(f"time hybrid cartesian mesh with all bnd : {cube_allbnd_h}")
    C.convertPyTree2File(t, 'cubeH_allbnd.hdf')

    # join+bnd
    # --------
    start_time = time.time()
    t = factory.cube_hybrid_join_bnd()
    cube_join_bnd_h = (time.time() - start_time)
    print(f"time hybrid cartesian mesh with join and bnd : {cube_join_bnd_h}")
    C.convertPyTree2File(t, 'cubeH_join_bnd.hdf')
    # sys.exit(1)
