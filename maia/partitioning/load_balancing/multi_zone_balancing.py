from mpi4py import MPI
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
    n_elem_zone_abs = {key:value for key,value in sorted_zone}
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
      ii_rank  = 0
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
          # if(ii_rank > n_rank):
          if(ii_rank > n_rank_zone):
            ii_rank = 0
          dproc_to_zone[j_zone].append(ii_rank+cur_rank)
          # LOG.debug(' '*8 + " ~> cur_rank : {0} ".format(cur_rank))
          # LOG.debug(' '*8 + " ~> ii_rank  : {0} ".format(ii_rank))
          # LOG.debug(' '*8 + " ~> Add r_elem : {0} to proc : {1} [{2}/{3}] ".format(r_elem, ii_rank+cur_rank, cur_rank, ii_rank))
          ii_rank += 1
          r_elem  -= j_elem
        # ooooooooooooooooooooooooooooooooooooooooo

      # *******************************************
      cur_rank += n_rank_zone
      # *******************************************
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Some zone have no proc affected and some rank are steal not assigned
    n_rank_remain  = n_rank - cur_rank
    d_remain_zones = {zone : j_elem for zone,j_elem in n_elem_zone_abs.items() if dproc_to_zone[zone] == []}

    # LOG.debug(' '*8 + " ~> Some zone have no proc affected and some rank are steal not assigned : {0}/{1} ".format(d_remain_zones, n_rank_remain))

    while d_remain_zones:
      # LOG.debug(' '*8 + " n_rank_remain : {0} ".format(n_rank_remain))

      n_elem_remain = 0
      for j_zone, j_elem in d_remain_zones.items():
        if (n_elem_remain + j_elem) <= mean_per_rank:
          dproc_to_zone[j_zone].append(cur_rank)
          n_elem_remain += j_elem
      # LOG.debug(' '*8 + " cur_rank : {0} ".format(cur_rank))
      cur_rank += 1

      n_rank_remain = n_rank-cur_rank
      d_remain_zones = {zone : j_elem for zone,j_elem in d_remain_zones.items() if dproc_to_zone[zone] == []}

      if n_rank_remain == 0:
        break
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Some zone have no proc affected and all ranks are assigned
    # LOG.debug(' '*8 + " Some zone have no proc affected and all ranks are assigned : {0}/{1} ".format(d_remain_zones, n_rank_remain))
    n_elem_mpi_tmp = np.zeros(n_rank, dtype=np.float64)
    for i_zone, lprocs in dproc_to_zone.items():
      n_elem_mpi_tmp[lprocs] += n_elem_per_zone[i_zone]/len(lprocs)

    while d_remain_zones:
      # *******************************************
      min_loaded_proc = np.argmin(n_elem_mpi_tmp)
      min_load        = n_elem_mpi_tmp[min_loaded_proc]
      #print "min_load = ", min_load, " et min_loaded_proc = ", min_loaded_proc
      # *******************************************

      # *******************************************
      cur_zone_to_add, cur_zone_to_add_n_elem = next(iter(d_remain_zones.items()))
      #print cur_zone_to_add, cur_zone_to_add_n_elem
      # *******************************************

      # *******************************************
      dproc_to_zone[cur_zone_to_add].append(min_loaded_proc)
      n_elem_mpi_tmp[min_loaded_proc] += cur_zone_to_add_n_elem
      # *******************************************

      # *******************************************
      d_remain_zones = {zone : j_elem for zone,j_elem in d_remain_zones.items() if dproc_to_zone[zone] == []}
      # *******************************************

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Some rank are steal not assigned
    # LOG.debug(' '*8 + " ~> Some rank are steal not assigned / cur_rank : {0} ".format(cur_rank))
    while(cur_rank != n_rank):
      zone_to_delem = {zone : n_elem_per_zone[zone]/len(lprocs) for zone, lprocs in dproc_to_zone.items()}
      maxZone = max(zone_to_delem.items(), key=lambda item:item[1])[0]

      # > Fill maxZone
      # LOG.debug(' '*8 + " ~> maxZone : {0} ".format(maxZone))
      # LOG.debug(' '*8 + " ~> cur_rank : {0} ".format(cur_rank))
      dproc_to_zone[maxZone].append(cur_rank)
      cur_rank += 1
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


def balance_with_non_uniform_weights(n_elem_per_zone, n_rank,
  tol_increment=0.01, n_elem_async_mpi=50000):
  """
  New balacing method. Assume that zone can be splitted in order to 
  obtain the prescripted number of elems in each part.
  Allow a proc to be affected to several zones.
  Input : n_elem_per_zone (dict) : number of cells in each zone
          n_rank (int)           : number of available ranks
  Options : tol_increment
            n_elem_async_mpi
  Output : repart_per_zone (dict) : for each zone, array of size
  n_rank indicating the number of cells affected to ith rank.

  May be usefull when weigthed partitioning is available
  """
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info(' '*2 + "================================================================= " )
  LOG.info(' '*2 + "================  Start Computing Distribution  ================= " )
  converged = False
  adjusted_tol = 0
  initial_allowed_load_pc = 0.75
  min_chunk_size_pc       = 0.20
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # > Sort zones by number of elements
  sorted_zone = sorted(n_elem_per_zone.items(), key=lambda item:item[1])
  n_elmt_tot = sum([val for (zone, val) in sorted_zone])
  # > Compute mean per rank
  initial_mean_per_rank = n_elmt_tot // n_rank
  mean_remainder     = n_elmt_tot  % n_rank
  # > Add one for the remainder
  if (mean_remainder) != 0:
    initial_mean_per_rank += 1

  while not converged:
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Init working dictionnary and loading array
    n_elem_zone_abs = {key:value for key,value in sorted_zone}
    proc_load = np.zeros(n_rank, dtype=np.int32)
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Compute new mean_per_rank and thresholds
    mean_per_rank      = int(initial_mean_per_rank * (1+adjusted_tol))
    allowed_load_ini   = int(round(initial_allowed_load_pc * mean_per_rank))
    min_chunk_size     = int(round(min_chunk_size_pc * mean_per_rank))
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Verbose
    LOG.info(' '*2 + "----------------------------  INPUT  ---------------------------- " )
    LOG.info(' '*4 + "nbTotalElem : {0} on {1} processors".format(n_elmt_tot, n_rank))
    LOG.info(' '*4 + "meanPerRank : {0} (remainder was {1})".format(mean_per_rank, mean_remainder))
    LOG.info(' '*4 + "iniMaxLoad  : {0}".format(allowed_load_ini))
    LOG.info(' '*4 + "minChunkSize: {0}".format(min_chunk_size))
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Loop over the zones and affect procs
    repart_per_zone = {z_name : n_rank*[0] for z_name in n_elem_zone_abs}
    current_proc = 0

    LOG.info(' '*2 + "--------------------------  PROCEDING  -------------------------- " )

    LOG.info(' '*4 + "Step 0. Try to combinate zones to reach meanPerRank")
    # > if mean_remainder != 0, we want to combinate zones to find mean_per_rank or mean_per_rank-1
    nb_of_search = 1 if mean_remainder == 0 else 2
    for k in range(nb_of_search):
      match_idx = True #To enter the loop
      while match_idx:
        searching_list = np.fromiter(n_elem_zone_abs.values(), dtype=np.int32)
        match_idx = search_match(searching_list, mean_per_rank-k, tol=0)
        if match_idx:
          z_names = [list(n_elem_zone_abs.keys())[idx] for idx in match_idx]
          LOG.debug(' '*4 + "~>Match found : affect zones {0} to proc {1}".format(z_names, current_proc))
          for z_name in z_names:
            n_elem = n_elem_zone_abs.pop(z_name)
            repart_per_zone[z_name][current_proc] += n_elem
            proc_load[current_proc] += n_elem
          current_proc += 1

    LOG.info(' '*4 + "Step 1. Use small zones to fill procs up to 75% of load")

    i_zone = 0
    n_elem_zone_abs_items = list(n_elem_zone_abs.items())
    while i_zone < len(n_elem_zone_abs_items):
      z_name, n_elem = n_elem_zone_abs_items[i_zone]
      # *******************************************
      available_size = allowed_load_ini - proc_load[current_proc]
      # > La zone est elle assez petite pour le proc en cours?
      if n_elem <= available_size:
        # LOG.debug(' '*4 + "~>affect zone {0} to proc {1}".format(z_name, current_proc))
        n_elem_zone_abs.pop(z_name)
        repart_per_zone[z_name][current_proc] += n_elem
        proc_load[current_proc] += n_elem
        i_zone += 1
      # > La zone est elle trop grande pour tous les procs ?
      elif n_elem > allowed_load_ini:
        # LOG.debug(' '*4 + "~>all small zones have been affected")
        break
      # > Reste il des procs libre pour prendre la zone ?
      elif current_proc < n_rank - 1:
        current_proc += 1
      else:
        # LOG.debug(' '*4 + "~>all procs are loaded to 75%")
        break

    LOG.info(' '*4 + "Step 2. Treat remaning zones, allowing cuts")
    is_fully_open = np.ones(n_rank, dtype=bool)
    for z_name in n_elem_zone_abs:
      n_elem = n_elem_zone_abs[z_name]
      # LOG.debug(' '*4 + "~>treat zone {0} of size {1}".format(z_name, n_elem))
      while n_elem > 0:
        # > Obtention des procs pas encore à 100%
        not_full_procs_idx  = np.where(proc_load < mean_per_rank-1)[0]
        not_full_procs_data = proc_load[not_full_procs_idx]
        # > Extraction des procs pas encore à 80%
        fully_open_procs_idx  = np.extract(is_fully_open[not_full_procs_idx], not_full_procs_idx)
        fully_open_procs_data = proc_load[fully_open_procs_idx]
        # > Recherche du proc le moins chargé, et du proc le plus chargé parmis ceux < 80%
        min_loaded_proc = not_full_procs_idx[np.argmin(not_full_procs_data)]
        if fully_open_procs_idx.size > 0:
          max_loaded_proc       = fully_open_procs_idx[np.argmax(fully_open_procs_data)]
          available_size_on_max = mean_per_rank - proc_load[max_loaded_proc]
        else:
          max_loaded_proc       = None
        # LOG.debug(' '*4 + "  n_elem = {0}, maxProc is {1} (load = {3}/{5}), minProc is {2} (load = {4}/{5})"
        #   .format(n_elem, max_loaded_proc, min_loaded_proc, proc_load[max_loaded_proc], proc_load[min_loaded_proc], mean_per_rank))
        n_elem_to_load = None
        loading_proc   = None

        # > Cas où un proc peut se compléter exactement
        matching_procs = np.where(proc_load + n_elem == mean_per_rank)[0]
        if len(matching_procs) > 0:
          # LOG.debug(' '*6 + "~>zone perfectly match with proc {0}".format(matching_procs[0]))
          n_elem_to_load = n_elem
          loading_proc   = matching_procs[0]

        # > Cas où la zone est devenue petite et peut rentrer sur le proc le moins chargé,
        # à condition de ne pas dépasser 75% (on revient au step 1)
        # avant -> if n_elem <= allowed_load_ini and proc_load[min_loaded_proc]==0:
        elif (n_elem + proc_load[min_loaded_proc]) <= allowed_load_ini:
          # LOG.debug(' '*6 + "~>zone is now small enought to fit in open proc {0}".format(min_loaded_proc))
          n_elem_to_load = n_elem
          loading_proc   = min_loaded_proc

        # > Cas où il existe un proc max et qu'il peut se compléter en laissant au moins 20% de la zone
        #   *et* n_elem > min_chunk_size
        #   "Ce que je peux charger est inférieur à ce que j'ai le droit de charger"
        elif (max_loaded_proc is not None) and (available_size_on_max <= (n_elem - min_chunk_size)):
          # LOG.debug(' '*6 + "~>complete proc {0} using {1} elems because remaining chunk will be >20%".format(
            # max_loaded_proc, available_size_on_max))
          n_elem_to_load = available_size_on_max
          loading_proc   = max_loaded_proc


        # > Cas où le proc max ne peut pas se compléter car la zone est trop grande,
        #   mais où celle ci peut etre coupée en deux partitions de taille > 20%.
        #   On cherche
        #   a. à donner le gros morceau au premier proc pouvant rester sous 80% après chargement
        #   b. à défaut, au proc qui sera le plus proche de la charge idéale.
        elif n_elem >= 2*min_chunk_size:
          find_proc    = False
          min_gap_proc = None
          min_gap      = 2**32

          n_elem_to_load = n_elem - min_chunk_size
          for iProc in not_full_procs_idx[np.argsort(not_full_procs_data)][::-1]:
            available_size_proc = mean_per_rank - proc_load[iProc]          # Place courante
            potential_load      = proc_load[iProc] + n_elem_to_load          # Charge si on prends la zone
            gap                 = mean_per_rank - potential_load
            if abs(gap) < min_gap:
              min_gap      = abs(gap)
              min_gap_proc = iProc

            # > Equivalent à n_elem_to_load <= available_size_proc - min_chunk_size
            # > This is case a
            if n_elem <= available_size_proc:
              find_proc    = True
              loading_proc = iProc
              # LOG.debug(' '*6 + "~>save a chunk of size 20% and load proc {0} (below 80%) "\
               # "with the remaining {1} elems".format(loading_proc, n_elem - min_chunk_size))
              break

          # > Execute case b. only if no break occured (ie case a did not happend)
          else:
            loading_proc = min_gap_proc
            is_fully_open[loading_proc] = False
            # LOG.debug(' '*6 + "~>save a chunk of size 20% and load proc {0} (above 80%, "\
             # "but least worst) with the remaining {1} elems".format(loading_proc, n_elem - min_chunk_size))


        # > La zone est < 2*min_chunk_size, mais tous les procs sont déjà chargés à > 75%
        else:
          n_available_procs = len(not_full_procs_idx)
          if n_available_procs > 1 and n_elem > 2*n_elem_async_mpi:
            n_elem_to_load1 = int(n_elem/2.)
            repart_per_zone[z_name][min_loaded_proc] += n_elem_to_load1
            proc_load[min_loaded_proc] += n_elem_to_load1
            n_elem_zone_abs[z_name] -= n_elem_to_load1
            # Careful - n_elem_to_load has already been added
            if mean_per_rank - proc_load[min_loaded_proc] < min_chunk_size:
              is_fully_open[min_loaded_proc] = False

            n_elem = n_elem - n_elem_to_load1
            loading_proc   = np.argmin(proc_load)
            n_elem_to_load = n_elem
            # LOG.debug(' '*6 + "~>zone can't be divided but there is no more open proc ; "\
             # "split between procs {0} and {1}".format(min_loaded_proc, loading_proc))
            if mean_per_rank - (proc_load[loading_proc] + n_elem_to_load) < min_chunk_size:
              is_fully_open[loading_proc] = False
          else:
            # LOG.debug(' '*6 + "~>zone can't be divided but there is no more open proc ; "\
             # "affect zone to min proc {0}".format(min_loaded_proc))
            n_elem_to_load = n_elem
            loading_proc   = min_loaded_proc
            if mean_per_rank - (proc_load[loading_proc] + n_elem_to_load) < min_chunk_size:
              is_fully_open[loading_proc] = False


        # > Effective load
        repart_per_zone[z_name][loading_proc] += n_elem_to_load
        proc_load[loading_proc] += n_elem_to_load
        n_elem_zone_abs[z_name] -= n_elem_to_load
        n_elem -= n_elem_to_load

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # > Restart if maxload is to high
    converged = np.max(proc_load) <= mean_per_rank
    if not converged:
      adjusted_tol += tol_increment
      LOG.info(' '*2 + '= '*33)
      LOG.info(' '*2 + "===== Poor balance -> Increase tolerance to {0:d} % and restart ===== ".format(
        int(100*adjusted_tol)))
      LOG.info(' '*2 + '= '*33)
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  LOG.info(' '*2 + "--------------------------- RUN SUMMARY -------------------------- " )
  LOG.info(' '*4 + "---> Number of runs       : {0}".format(int(adjusted_tol/tol_increment)+1))
  LOG.info(' '*4 + "---> Final tolerance used : {0}%".format(int(100*adjusted_tol)))
  LOG.info(' '*2 + 66*"-")
  # > Check balance
  #computeBalanceAndSplits(repart_per_zone, n_rank)
  #rMean, rRMS, rMini, rMaxi, rmspc = computeLoadBalance(nbLocalElemPerZone, repart_per_zone, n_rank)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  LOG.info(' '*2 + "=================  End Computing Distribution  ================= " )
  LOG.info(' '*2 + "================================================================ " )
  return repart_per_zone
