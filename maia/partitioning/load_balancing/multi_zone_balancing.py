from mpi4py import MPI
import collections as CLT
import logging     as LOG
import Converter.Internal as I
import numpy as np
from .utils import search_match
from maia.partitioning.load_balancing import single_zone_balancing

def balance_with_uniform_weights(n_elem_per_zone, n_rank):
    """
    Old balacing method. After the procs are affected to the zones,
    each zone is shared in equally sized parts.
    Allow a proc to be affected to several zones.
    Input : n_elem_per_zone (dict) : number of cells in each zone
            n_rank (int)           : number of available ranks
    Output : repart_per_zone (dict) : for each zone, array of size
    n_rank indicating the number of cells affected to ith rank.

    May be usefull when weigthed partitioning is not available
    """
    # LOG.info(' '*4 + " ~> PrepareDistribBaseGen ")

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    sorted_zone = sorted(n_elem_per_zone.items(), key=lambda item:item[1], reverse=True)
    n_elem_zone_abs = CLT.OrderedDict()
    for zone in sorted_zone:
      n_elem_zone_abs[zone[0]] = zone[1]
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # I/ Step 1
    # > Compute mean per rank
    n_elem_tot    = sum(n_elem_zone_abs.values())
    mini_per_rank = min(n_elem_zone_abs.values())
    maxi_per_rank = max(n_elem_zone_abs.values())
    mean_per_rank = n_elem_tot//n_rank
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Verbose
    # LOG.debug(' '*8 + "*"*100)
    # LOG.debug(' '*8 + " n_elem_tot : {0} on {1} processors".format(n_elem_tot, n_rank))
    # LOG.debug(' '*8 + " mean_per_rank : {0} ".format(mean_per_rank))
    # LOG.debug(' '*8 + " mini_per_rank : {0} ".format(mini_per_rank))
    # LOG.debug(' '*8 + " maxi_per_rank : {0} ".format(maxi_per_rank))
    # LOG.debug(' '*8 + "*"*100)
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Init the dictionnary containing procs list
    dproc_to_zone = {key : [] for key in n_elem_zone_abs}
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    cur_rank = 0
    for i_zone, i_elem in n_elem_zone_abs.items():
      # *******************************************
      # > Verbose
      # LOG.debug(' '*8 + "-"*100)
      # LOG.debug(' '*8 + "~> i_zone/i_elem : {0}/{1} ".format(i_zone, i_elem ))
      # LOG.debug(' '*8 + "~> cur_rank : {0} ".format(cur_rank))
      # *******************************************

      # *******************************************
      # > Compute n_rank affected to the current Zone
      n_rank_zone = i_elem//mean_per_rank
      # print("n_rank_zone = {}".format(n_rank_zone))
      # LOG.debug(' '*8 + " ~> n_rank_zone : {0} ".format(n_rank_zone))
      # *******************************************

      # *******************************************
      for iproc in range(n_rank_zone):
        dproc_to_zone[i_zone].append(iproc+cur_rank)
      # *******************************************

      # *******************************************
      # > Compute the number of remaining (rest of integer division )
      r_elem = i_elem - n_rank_zone*mean_per_rank
      # LOG.debug(' '*8 + " ~> r_elem : {0} ".format(r_elem))
      # *******************************************

      # *******************************************
      # > Loop on the next Zone
      iiRank  = 0
      for j_zone, j_elem in n_elem_zone_abs.items():
        # ooooooooooooooooooooooooooooooooooooooooo
        # > Verbose
        LOG.debug(' '*20 + "o"*80 )
        LOG.debug(' '*20 + "~> j_zone/j_elem : {0}/{1} ".format(j_zone, j_elem))
        # ooooooooooooooooooooooooooooooooooooooooo

        # ooooooooooooooooooooooooooooooooooooooooo
        # > Check if j_zone have proc already affected
        if(len(dproc_to_zone[j_zone]) != 0):
          continue # > Zone already affectted
        # ooooooooooooooooooooooooooooooooooooooooo

        # ooooooooooooooooooooooooooooooooooooooooo
        # > Check if no proc to affect
        if(n_rank_zone == 0):
          continue # > No proc to affect
        # ooooooooooooooooooooooooooooooooooooooooo

        # ooooooooooooooooooooooooooooooooooooooooo
        # > Test if j_elem < r_elem
        if(j_elem <= r_elem):
          # if(iiRank > n_rank):
          if(iiRank > n_rank_zone):
            iiRank = 0
          dproc_to_zone[j_zone].append(iiRank+cur_rank)
          # LOG.debug(' '*8 + " ~> cur_rank : {0} ".format(cur_rank))
          # LOG.debug(' '*8 + " ~> iiRank  : {0} ".format(iiRank))
          # LOG.debug(' '*8 + " ~> Add r_elem : {0} to proc : {1} [{2}/{3}] ".format(r_elem, iiRank+cur_rank, cur_rank, iiRank))
          iiRank += 1
          r_elem  -= j_elem
        # ooooooooooooooooooooooooooooooooooooooooo

      # *******************************************
      cur_rank += n_rank_zone
      # *******************************************
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Some zone have no proc affected and some rank are steal not assigned
    n_rank_remain  = n_rank-cur_rank
    dRemainZones = CLT.OrderedDict()

    for j_zone, j_elem in n_elem_zone_abs.items():
      if(len(dproc_to_zone[j_zone]) == 0):
        dRemainZones[j_zone] = j_elem

    # LOG.debug(' '*8 + " ~> Some zone have no proc affected and some rank are steal not assigned : {0}/{1} ".format(dRemainZones, n_rank_remain))

    while len(dRemainZones)!= 0:
      # LOG.debug(' '*8 + " n_rank_remain : {0} ".format(n_rank_remain))

      nSumElemRemain = 0
      for j_zone, j_elem in dRemainZones.items():
        if (nSumElemRemain + j_elem) <= mean_per_rank:
          dproc_to_zone[j_zone].append(cur_rank)
          nSumElemRemain += j_elem
      # LOG.debug(' '*8 + " cur_rank : {0} ".format(cur_rank))
      cur_rank += 1

      n_rank_remain = n_rank-cur_rank
      dRemainZonesBis = CLT.OrderedDict()

      for j_zoneRemain, j_elem_remain in dRemainZones.items():
        if(len(dproc_to_zone[j_zoneRemain]) == 0):
          dRemainZonesBis[j_zoneRemain] = j_elem_remain

      dRemainZones = dRemainZonesBis

      if n_rank_remain == 0:
        break
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Some zone have no proc affected and all ranks are assigned
    # LOG.debug(' '*8 + " Some zone have no proc affected and all ranks are assigned : {0}/{1} ".format(dRemainZones, n_rank_remain))
    nElemMPITmp = CLT.OrderedDict()
    for iRank in range(n_rank):
      nElemMPITmp[iRank] = 0

    for i_zone, lprocs in dproc_to_zone.items():
      for iRank in lprocs:
        nElemMPITmp[iRank] += n_elem_per_zone[i_zone]/len(lprocs)

    while len(dRemainZones)!= 0:

      # *******************************************
      ProcWithMinLoad = next(iter(nElemMPITmp.items()))[0]
      MinLoad         = next(iter(nElemMPITmp.items()))[1]
      #print "MinLoad = ", MinLoad, " et ProcWithMinLoad = ", ProcWithMinLoad
      for iRank in nElemMPITmp.keys():
        #print iRank, nElemMPITmp[iRank]
        if nElemMPITmp[iRank] < MinLoad:
          #print "inferior"
          MinLoad = nElemMPITmp[iRank]
          ProcWithMinLoad = iRank
          #print " > NEW : MinLoad = ", MinLoad, " et ProcWithMinLoad = ", ProcWithMinLoad
      # *******************************************

      # *******************************************
      curZoneToAdded      = next(iter(dRemainZones.items()))[0]
      nElemCurZoneToAdded = next(iter(dRemainZones.items()))[1]
      # *******************************************
      #print curZoneToAdded, nElemCurZoneToAdded

      # *******************************************
      dproc_to_zone[curZoneToAdded].append(ProcWithMinLoad)
      # *******************************************

      # *******************************************
      dRemainZonesBis = CLT.OrderedDict()
      for jZoneRemain, jElemRemain in dRemainZones.items():
        if(len(dproc_to_zone[jZoneRemain]) == 0):
          dRemainZonesBis[jZoneRemain] = jElemRemain
      dRemainZones = dRemainZonesBis
      # *******************************************

      #print "ProcWithMinLoad = ",ProcWithMinLoad, " et nElemMPITmp[ProcWithMinLoad] = ", nElemMPITmp[ProcWithMinLoad]

      # *******************************************
      nElemMPITmp[ProcWithMinLoad] += nElemCurZoneToAdded
      # *******************************************

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Some rank are steal not assigned
    # LOG.debug(' '*8 + " ~> Some rank are steal not assigned / cur_rank : {0} ".format(cur_rank))
    while(cur_rank != n_rank):
      # *******************************************
      ZoneWithMaxSize = -1
      for i_zone, lprocs in dproc_to_zone.items():
        dElem = n_elem_per_zone[i_zone]/len(lprocs)
        ZoneWithMaxSize = max(ZoneWithMaxSize, dElem)
        if(ZoneWithMaxSize == dElem):
          maxZone = i_zone
      # *******************************************

      # *******************************************
      # > Fill maxZone
      # LOG.debug(' '*8 + " ~> maxZone : {0} ".format(maxZone))
      # LOG.debug(' '*8 + " ~> cur_rank : {0} ".format(cur_rank))
      dproc_to_zone[maxZone].append(cur_rank)
      cur_rank += 1
      # *******************************************
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Remove double entry
    for i_zone in dproc_to_zone.keys():
      dproc_to_zone[i_zone] = list(set(dproc_to_zone[i_zone]))
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Format
    repart_per_zone = dict()
    for zone, procs in dproc_to_zone.items():
      homonegeous_split = single_zone_balancing.homogeneous_repart(n_elem_per_zone[zone], len(procs))
      # > Unpack
      homonegeous_split_it = iter(homonegeous_split)
      repart_per_zone[zone] = [next(homonegeous_split_it) if i in procs else 0 for i in range(n_rank)]
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    return repart_per_zone


def balance_with_non_uniform_weights(nElemPerZone, nRank,
  tolIncrement=0.01, nElemToRecoverMPIExchange=50000):
  """
  New balacing method. Assume that zone can be splitted in order to 
  obtain the prescripted number of elems in each part.
  Allow a proc to be affected to several zones.
  Input : n_elem_per_zone (dict) : number of cells in each zone
          n_rank (int)           : number of available ranks
  Options : tolIncrement
            nElemToRecoverMPIExchange
  Output : repart_per_zone (dict) : for each zone, array of size
  n_rank indicating the number of cells affected to ith rank.

  May be usefull when weigthed partitioning is available
  """
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info(' '*2 + "================================================================= " )
  LOG.info(' '*2 + "================  Start Computing Distribution  ================= " )
  converged = False
  adjustedTol = 0
  initialAllowedLoadPercent = 0.75
  minChunkSizePercent       = 0.20
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # > Sort zones by number of elements
  SortedZone = sorted(nElemPerZone.items(), key=lambda item:item[1])
  nTElem = sum([val for (zone, val) in SortedZone])
  # > Compute mean per rank
  initialMeanPerRank = nTElem/nRank
  meanRemainder      = nTElem % nRank
  # > Add one for the remainder
  if (meanRemainder) != 0:
    initialMeanPerRank += 1

  while not converged:
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Init working dictionnary and loading array
    nElemZoneAbs = CLT.OrderedDict()
    for iZone in SortedZone:
      nElemZoneAbs[iZone[0]] = iZone[1]
    procLoad = np.zeros(nRank, dtype=np.int32)
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Compute new meanPerRank and thresholds
    meanPerRank        = int(initialMeanPerRank * (1+adjustedTol))
    initialAllowedLoad = int(round(initialAllowedLoadPercent * meanPerRank))
    minChunkSize       = int(round(minChunkSizePercent * meanPerRank))
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Verbose
    LOG.info(' '*2 + "----------------------------  INPUT  ---------------------------- " )
    LOG.info(' '*4 + "nbTotalElem : {0} on {1} processors".format(nTElem, nRank))
    LOG.info(' '*4 + "meanPerRank : {0} (remainder was {1})".format(meanPerRank, meanRemainder))
    LOG.info(' '*4 + "iniMaxLoad  : {0}".format(initialAllowedLoad))
    LOG.info(' '*4 + "minChunkSize: {0}".format(minChunkSize))
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Loop over the zones and affect procs
    repart_per_zone = {z_name : nRank*[0] for z_name in nElemZoneAbs}
    currentProc = 0

    LOG.info(' '*2 + "--------------------------  PROCEDING  -------------------------- " )

    LOG.info(' '*4 + "Step 0. Try to combinate zones to reach meanPerRank")
    # > if meanRemainder != 0, we want to combinate zones to find meanPerRank or meanPerRank-1
    nbOfSearch = 1 if meanRemainder == 0 else 2
    for k in range(nbOfSearch):
      matchIdx = True #To enter the loop
      while matchIdx:
        searchingList = np.fromiter(nElemZoneAbs.values(), dtype=np.int32)
        matchIdx = search_match(searchingList, meanPerRank-k, tol=0)
        if matchIdx:
          zNames = [list(nElemZoneAbs.keys())[idx] for idx in matchIdx]
          LOG.debug(' '*4 + "~>Match found : affect zones {0} to proc {1}".format(zNames, currentProc))
          for zName in zNames:
            nElem = nElemZoneAbs.pop(zName)
            repart_per_zone[zName][currentProc] += nElem
            procLoad[currentProc] += nElem
          currentProc += 1

    LOG.info(' '*4 + "Step 1. Use small zones to fill procs up to 75% of load")

    iZone = 0
    nElemZoneAbsItems = list(nElemZoneAbs.items())
    while iZone < len(nElemZoneAbsItems):
      zName, nElem = nElemZoneAbsItems[iZone]
      # *******************************************
      availableSize = initialAllowedLoad - procLoad[currentProc]
      # > La zone est elle assez petite pour le proc en cours?
      if nElem <= availableSize:
        # LOG.debug(' '*4 + "~>affect zone {0} to proc {1}".format(zName, currentProc))
        nElemZoneAbs.pop(zName)
        repart_per_zone[zName][currentProc] += nElem
        procLoad[currentProc] += nElem
        iZone += 1
      # > La zone est elle trop grande pour tous les procs ?
      elif nElem > initialAllowedLoad:
        # LOG.debug(' '*4 + "~>all small zones have been affected")
        break
      # > Reste il des procs libre pour prendre la zone ?
      elif currentProc < nRank - 1:
        currentProc += 1
      else:
        # LOG.debug(' '*4 + "~>all procs are loaded to 75%")
        break

    LOG.info(' '*4 + "Step 2. Treat remaning zones, allowing cuts")
    isFullyOpen = np.ones(nRank, dtype=bool)
    for zName in nElemZoneAbs:
      nElem = nElemZoneAbs[zName]
      # LOG.debug(' '*4 + "~>treat zone {0} of size {1}".format(zName, nElem))
      while nElem > 0:
        # > Obtention des procs pas encore à 100%
        notFullProcsIdx  = np.where(procLoad < meanPerRank-1)[0]
        notFullProcsData = procLoad[notFullProcsIdx]
        # > Extraction des procs pas encore à 80%
        fullyOpenProcsIdx  = np.extract(isFullyOpen[notFullProcsIdx], notFullProcsIdx)
        fullyOpenProcsData = procLoad[fullyOpenProcsIdx]
        # > Recherche du proc le moins chargé, et du proc le plus chargé parmis ceux < 80%
        minLoadedProc = notFullProcsIdx[np.argmin(notFullProcsData)]
        if fullyOpenProcsIdx.size > 0:
          maxLoadedProc      = fullyOpenProcsIdx[np.argmax(fullyOpenProcsData)]
          availableSizeOnMax = meanPerRank - procLoad[maxLoadedProc]
        else:
          maxLoadedProc      = None
        # LOG.debug(' '*4 + "  nElem = {0}, maxProc is {1} (load = {3}/{5}), minProc is {2} (load = {4}/{5})"
        #   .format(nElem, maxLoadedProc, minLoadedProc, procLoad[maxLoadedProc], procLoad[minLoadedProc], meanPerRank))
        nElemToLoad = None
        loadingProc = None

        # > Cas où un proc peut se compléter exactement
        matchingProcs = np.where(procLoad + nElem == meanPerRank)[0]
        if len(matchingProcs) > 0:
          # LOG.debug(' '*6 + "~>zone perfectly match with proc {0}".format(matchingProcs[0]))
          nElemToLoad = nElem
          loadingProc = matchingProcs[0]

        # > Cas où la zone est devenue petite et peut rentrer sur le proc le moins chargé,
        # à condition de ne pas dépasser 75% (on revient au step 1)
        # avant -> if nElem <= initialAllowedLoad and procLoad[minLoadedProc]==0:
        elif (nElem + procLoad[minLoadedProc]) <= initialAllowedLoad:
          # LOG.debug(' '*6 + "~>zone is now small enought to fit in open proc {0}".format(minLoadedProc))
          nElemToLoad = nElem
          loadingProc = minLoadedProc

        # > Cas où il existe un proc max et qu'il peut se compléter en laissant au moins 20% de la zone
        #   *et* nElem > minChunkSize
        #   "Ce que je peux charger est inférieur à ce que j'ai le droit de charger"
        elif (maxLoadedProc is not None) and (availableSizeOnMax <= (nElem - minChunkSize)):
          # LOG.debug(' '*6 + "~>complete proc {0} using {1} elems because remaining chunk will be >20%".format(
            # maxLoadedProc, availableSizeOnMax))
          nElemToLoad = availableSizeOnMax
          loadingProc = maxLoadedProc


        # > Cas où le proc max ne peut pas se compléter car la zone est trop grande,
        #   mais où celle ci peut etre coupée en deux partitions de taille > 20%.
        #   On cherche
        #   a. à donner le gros morceau au premier proc pouvant rester sous 80% après chargement
        #   b. à défaut, au proc qui sera le plus proche de la charge idéale.
        elif nElem >= 2*minChunkSize:
          findProc   = False
          minGapProc = None
          minGap     = 2**32

          nElemToLoad = nElem - minChunkSize
          for iProc in notFullProcsIdx[np.argsort(notFullProcsData)][::-1]:
            availableSizeProc = meanPerRank - procLoad[iProc]          # Place courante
            potentialLoad     = procLoad[iProc] + nElemToLoad          # Charge si on prends la zone
            gap               = meanPerRank - potentialLoad
            if abs(gap) < minGap:
              minGap     = abs(gap)
              minGapProc = iProc

            # > Equivalent à nElemToLoad <= availableSizeProc - minChunkSize
            # > This is case a
            if nElem <= availableSizeProc:
              findProc = True
              loadingProc = iProc
              # LOG.debug(' '*6 + "~>save a chunk of size 20% and load proc {0} (below 80%) "\
               # "with the remaining {1} elems".format(loadingProc, nElem - minChunkSize))
              break

          # > Execute case b. only if no break occured (ie case a did not happend)
          else:
            loadingProc = minGapProc
            isFullyOpen[loadingProc] = False
            # LOG.debug(' '*6 + "~>save a chunk of size 20% and load proc {0} (above 80%, "\
             # "but least worst) with the remaining {1} elems".format(loadingProc, nElem - minChunkSize))


        # > La zone est < 2*minChunkSize, mais tous les procs sont déjà chargés à > 75%
        else:
          nbAvailableProcs = len(notFullProcsIdx)
          if nbAvailableProcs > 1 and nElem > 2*nElemToRecoverMPIExchange:
            nElemToLoad1 = int(nElem/2.)
            repart_per_zone[zName][minLoadedProc] += nElemToLoad1
            procLoad[minLoadedProc] += nElemToLoad1
            nElemZoneAbs[zName] -= nElemToLoad1
            # Careful - nElemToLoad has already been added
            if meanPerRank - procLoad[minLoadedProc] < minChunkSize:
              isFullyOpen[minLoadedProc] = False

            nElem = nElem - nElemToLoad1
            loadingProc = np.argmin(procLoad)
            nElemToLoad = nElem
            # LOG.debug(' '*6 + "~>zone can't be divided but there is no more open proc ; "\
             # "split between procs {0} and {1}".format(minLoadedProc, loadingProc))
            if meanPerRank - (procLoad[loadingProc] + nElemToLoad) < minChunkSize:
              isFullyOpen[loadingProc] = False
          else:
            # LOG.debug(' '*6 + "~>zone can't be divided but there is no more open proc ; "\
             # "affect zone to min proc {0}".format(minLoadedProc))
            nElemToLoad = nElem
            loadingProc = minLoadedProc
            if meanPerRank - (procLoad[loadingProc] + nElemToLoad) < minChunkSize:
              isFullyOpen[loadingProc] = False


        # > Effective load
        repart_per_zone[zName][loadingProc] += nElemToLoad
        procLoad[loadingProc] += nElemToLoad
        nElemZoneAbs[zName] -= nElemToLoad
        nElem -= nElemToLoad

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Restart if maxload is to high
    converged = np.max(procLoad) <= meanPerRank
    if not converged:
      adjustedTol += tolIncrement
      LOG.info(' '*2 + '= '*33)
      LOG.info(' '*2 + "===== Poor balance -> Increase tolerance to {0:d} % and restart ===== ".format(
        int(100*adjustedTol)))
      LOG.info(' '*2 + '= '*33)
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info(' '*2 + "--------------------------- RUN SUMMARY -------------------------- " )
  LOG.info(' '*4 + "---> Number of runs       : {0}".format(int(adjustedTol/tolIncrement)+1))
  LOG.info(' '*4 + "---> Final tolerance used : {0}%".format(int(100*adjustedTol)))
  LOG.info(' '*2 + 66*"-")
  # > Check balance
  #computeBalanceAndSplits(repart_per_zone, nRank)
  #rMean, rRMS, rMini, rMaxi, rmspc = computeLoadBalance(nbLocalElemPerZone, repart_per_zone, nRank)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  LOG.info(' '*2 + "=================  End Computing Distribution  ================= " )
  LOG.info(' '*2 + "================================================================ " )
  return repart_per_zone
