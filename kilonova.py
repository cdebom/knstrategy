class KNCalc():
    def __init__(self,
                 distance,
                 distance_err,
                 time_delay,
                 loglan=None,
                 vk=None,
                 logmass=None,
                 info_file="knsed_info.txt",
                 kn_weight_type="gaussian",
                 plot=False,
                 use_map="",
                 area_covered=0.9,
                 reduce_mapresolution=False,
                 set_mjd_correction=False,
                 m_exp_kncalc=False,
                 deep_coverage=0.5):

        # resolution=0.3
        self.sw_mexp = False
        self.distance = distance
        self.distance_err = distance_err

        self.loglan = loglan
        self.vk = vk
        self.logmass = logmass

        self.set_mjd_correction = set_mjd_correction

        if float(time_delay) > 400.8:
            MSG = "Currently, only times delays less thant 400.8 hours " +\
                  "(16.7 days) post merger are supported."

            print(MSG)
            sys.exit()

        self.delta_mjd = round(float(time_delay) / 24.0, 1)
        delta_mjd_full = float(time_delay) / 24.0

        delta_mjd_later = round(self.delta_mjd+0.3, 1)

        if (delta_mjd_full != self.delta_mjd) and (self.set_mjd_correction):
            self.set_mjd_correction = True
            if delta_mjd_full > self.delta_mjd:
                delta_mjd_corr = self.delta_mjd+0.1
                if (delta_mjd_full-self.delta_mjd) >= 0.05:
                    delta_mjd_later = self.delta_mjd+0.4
            else:
                delta_mjd_corr = self.delta_mjd-0.1
            delta_mjd_corr = round(delta_mjd_corr, 1)
        else:
            self.set_mjd_correction = False

        # Set directory for lookup table
        knlc_dir = os.getenv("DESGW_DIR", "./")
        if knlc_dir != "./":
            knlc_dir = knlc_dir + "/knlc/"

        # Choose lookup table based on time_delay
        if delta_mjd_later < 2.3:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry.csv')
        elif delta_mjd_later < 4.7:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry_2.csv')
        elif delta_mjd_later < 7.1:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry_3.csv')
        elif delta_mjd_later < 9.5:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry_4.csv')
        elif delta_mjd_later < 11.9:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry_5.csv')
        elif delta_mjd_later < 14.3:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry_6.csv')
        elif delta_mjd_later < 16.7:
            df_later = pd.read_csv(knlc_dir+'data/grouped_photometry_7.csv')

        if self.delta_mjd < 2.3:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry.csv')
        elif self.delta_mjd < 4.7:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry_2.csv')
        elif self.delta_mjd < 7.1:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry_3.csv')
        elif self.delta_mjd < 9.5:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry_4.csv')
        elif self.delta_mjd < 11.9:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry_5.csv')
        elif self.delta_mjd < 14.3:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry_6.csv')
        elif self.delta_mjd < 16.7:
            df = pd.read_csv(knlc_dir+'data/grouped_photometry_7.csv')

        if self.set_mjd_correction:
            if delta_mjd_corr < 2.3:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry.csv')
            elif delta_mjd_corr < 4.7:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry_2.csv')
            elif delta_mjd_corr < 7.1:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry_3.csv')
            elif delta_mjd_corr < 9.5:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry_4.csv')
            elif delta_mjd_corr < 11.9:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry_5.csv')
            elif delta_mjd_corr < 14.3:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry_6.csv')
            elif delta_mjd_corr < 16.7:
                df_corr = pd.read_csv(knlc_dir+'data/grouped_photometry_7.csv')

        df_later['ZMEAN'] = np.mean(df_later[['ZMIN', 'ZMAX']].values, axis=1)
        df['ZMEAN'] = np.mean(df[['ZMIN', 'ZMAX']].values, axis=1)

        mean_z = z_at_value(cosmo.luminosity_distance,
                            float(self.distance) * u.Mpc)
        template_df_mean = df[(df['ZMIN'].values < mean_z) &\
                              (df['ZMAX'].values > mean_z) &\
                              (df['DELTA_MJD'].values == self.delta_mjd)]
        template_df_mean = template_df_mean.copy().reset_index(drop=True)

        sed_filenames, kn_inds, vks, loglans, logmass_s = open_ascii_cat(
                                                                    info_file,
                                                                    unpack=True
                                                                )
        loglans = np.array(loglans).astype('float')
        kn_inds = np.array(kn_inds).astype('float')
        vks = np.array(vks).astype('float')
        logmass_s = np.array(logmass_s).astype('float')

        if template_df_mean.empty:
            checkzmax = df['ZMAX'].values > mean_z

            if not all(checkzmax):
                print(f"Object is too far away. mean_z is {mean_z}. Exiting.")
                self.Flag = 0
                return
            else:
                print("Something wrong with knlc photometry template library. Exiting")
                sys.exit()

        template_df_mean['WEIGHT'] = 1.0 / template_df_mean.shape[0]
        self.template_df_mean = template_df_mean

        # Full distance calculation
        template_df_full = df[df['DELTA_MJD'].values ==
                              self.delta_mjd].copy().reset_index(drop=True)
        template_df_later = df_later[df_later['DELTA_MJD'].values == delta_mjd_later].copy(
        ).reset_index(drop=True)

        if self.set_mjd_correction == True:

            template_df_corr = df_corr[df_corr['DELTA_MJD'].values == float(
                delta_mjd_corr)].copy().reset_index(drop=True)
            mag_g = template_df_full['MAG_g'].values
            mag_r = template_df_full['MAG_r'].values
            mag_i = template_df_full['MAG_i'].values
            mag_z = template_df_full['MAG_z'].values

            mag_g_corr = template_df_corr['MAG_g'].values
            mag_r_corr = template_df_corr['MAG_r'].values
            mag_i_corr = template_df_corr['MAG_i'].values
            mag_z_corr = template_df_corr['MAG_z'].values

            if delta_mjd_corr > self.delta_mjd:
                print("correcting magnitudes >")
                template_df_full['MAG_g'] = [
                    np.interp(delta_mjd_full,
                              [self.delta_mjd,delta_mjd_corr],
                              [mag_g[l], mag_g_corr[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_g))
                ]
                template_df_full['MAG_r'] = [
                    np.interp(delta_mjd_full,
                              [self.delta_mjd, delta_mjd_corr],
                              [mag_r[l], mag_r_corr[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_r))
                ]
                template_df_full['MAG_i'] = [
                    np.interp(delta_mjd_full,
                              [self.delta_mjd, delta_mjd_corr],
                              [mag_i[l], mag_i_corr[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_i))
                ]
                template_df_full['MAG_z'] = [
                    np.interp(delta_mjd_full,
                              [self.delta_mjd, delta_mjd_corr],
                              [mag_z[l], mag_z_corr[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_z))
                ]

            if delta_mjd_corr < self.delta_mjd:
                print("correcting magnitudes <")
                print(mag_g[0])
                template_df_full['MAG_g'] = [
                    np.interp(delta_mjd_full,
                              [delta_mjd_corr, self.delta_mjd],
                              [mag_g_corr[l], mag_g[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_g))
                ]
                template_df_full['MAG_r'] = [
                    np.interp(delta_mjd_full,
                              [delta_mjd_corr, self.delta_mjd],
                              [mag_r_corr[l], mag_r[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_r))
                    ]
                template_df_full['MAG_i'] = [
                    np.interp(delta_mjd_full,
                             [delta_mjd_corr, self.delta_mjd],
                             [mag_i_corr[l], mag_i[l]],
                             left=0,
                             right=0,
                             period=None) for l in range(0, len(mag_i))
                ]
                template_df_full['MAG_z'] = [
                    np.interp(delta_mjd_full,
                              [delta_mjd_corr, self.delta_mjd],
                              [mag_z_corr[l], mag_z[l]],
                              left=0,
                              right=0,
                              period=None) for l in range(0, len(mag_z))
                    ]

        loglans_full = []
        vks_full = []
        logmass_full = []
        for i in range(0, len(template_df_full['SIM_TEMPLATE_INDEX'])):
            _id = template_df_full['SIM_TEMPLATE_INDEX'][i]
            if _id == 27.5:
                _id = 99
                template_df_full['SIM_TEMPLATE_INDEX'][i] = 99
                template_df_later['SIM_TEMPLATE_INDEX'][i] = 99
            #print (_id)
            loglans_full.append(loglans[np.where(kn_inds == _id)[0]][0])
            vks_full.append(vks[np.where(kn_inds == _id)[0]][0])
            logmass_full.append(logmass_s[np.where(kn_inds == _id)[0]][0])

        loglans_full = np.array(loglans_full)
        vks_full = np.array(vks_full)
        logmass_full = np.array(logmass_full)
        mass_full = 10 ** logmass_full
        mass_ = 10 ** logmass_s
        preprocessed_weights = False
        if use_map != "":
            map_id = use_map.split("/")[-1]
            path2map = use_map.rstrip(map_id)
            map_id = map_id.rstrip(".fits")
            map_id = map_id.rstrip(".fits.gz")
            use_map_info = path2map+"lowres/"+map_id+"_mapinfo.npy"
            if m_exp_kncalc == False:
                use_map_weights = path2map+"weights/" + \
                    map_id+"ac"+str(area_covered)+".npy"
                use_map_weights_info = path2map+"weights/" + \
                    map_id+"ac"+str(area_covered)+"info.npy"
                try:

                    weights_pre = np.load(use_map_weights)
                    use_map = ""
                    print("Using preprocessed weights for ", use_map_weights)
                    preprocessed_weights = True
                    area_deg_info, resolution = np.load(use_map_weights_info)
                    self.Resolution = resolution
                    self.area_deg = area_deg_info  # num_pix_covered*resolution

                except:
                    print("Failed to load preprocessed weights for ",
                          use_map_weights)
            else:
                use_map_weights = path2map+"weights/"+map_id+"ac" + \
                    str(area_covered)+"ad"+str(deep_coverage)+".npy"
                use_map_weights_info = path2map+"weights/"+map_id+"ac" + \
                    str(area_covered)+"ad"+str(deep_coverage)+"info.npy"

                try:
                    weights_pre, weights_pre_deep = np.load(use_map_weights)
                    area_deg_info, area_deg_deep_info, resolution = np.load(
                        use_map_weights_info
                    )
                    self.area_deg_deep = area_deg_deep_info  # num_pix_covered_deep*resolution
                    self.Resolution = resolution
                    self.area_deg = area_deg_info  # num_pix_covered*resolution
                    # FIXME[num_pix_covered*resolution,num_pix_covered_deep*resolution,resolution]
                    use_map = ""
                    print("Using preprocessed weights for ", use_map_weights)
                    preprocessed_weights = True
                except:
                    print("Failed to load preprocessed weights for ",
                          use_map_weights)

        if use_map != "":
            print("Using maps, not the distance")
            use_map_lowres = use_map.split("/")[-1]
            path2map = use_map.rstrip(use_map_lowres)
            use_map_lowres = path2map+"lowres/"+use_map_lowres

            # try:
            print("Tryng to open low resolution map")
            try:
                pb, distmu, distsigma = hp.read_map(use_map_lowres, field=range(
                    3), verbose=False, dtype=[np.float64, np.float64, np.float64])
                distmu_hr_average, distmu_std, distsigma_hr_average, distsigma_std = np.load(
                    use_map_info)
                reduce_mapresolution = False
                read_lowres_map = True
            except:
                print(
                    "Failed to open low resolution map, opening high resolution map", use_map_lowres)
                pb, distmu, distsigma, distnorm = hp.read_map(use_map, field=range(4), verbose=False, dtype=[
                                                              np.float64, np.float64, np.float64, np.float64])  # clecio dtype='numpy.float64'
                read_lowres_map = False

            pb_check = pb[np.logical_not(np.isinf(distmu))]
            distsigma_check = distsigma[np.logical_not(np.isinf(distmu))]
            distmu_check = distmu[np.logical_not(np.isinf(distmu))]

            pb_check = pb_check[np.logical_not(np.isinf(distsigma_check))]
            distmu_check = distmu_check[np.logical_not(
                np.isinf(distsigma_check))]
            distsigma_check = distsigma_check[np.logical_not(
                np.isinf(distsigma_check))]

            distmu_check_average = np.average(distmu_check, weights=pb_check)
            distsigma_check_average = np.average(
                distsigma_check, weights=pb_check)
            try:
                mean_z = z_at_value(cosmo.luminosity_distance, float(
                    distmu_check_average) * u.Mpc)
            except:
                print('Warning: Object too close with distance ',
                      str(distmu_check_average))
                self.Flag = 0
                return

            mean_z68 = z_at_value(cosmo.luminosity_distance, float(
                distmu_check_average+distsigma_check_average) * u.Mpc)
            z_max = max(template_df_full['ZMEAN'].values)
            luminosity_dist_max = max(cosmo.luminosity_distance(
                np.unique(template_df_full['ZMEAN'].values)))

            print(mean_z)
            print(mean_z68)
            print(z_max)
            if float(mean_z68) > float(z_max):
                print('Warning: Object too far away zmax= ',
                      str(z_max), ' z_event ', str(mean_z))
                self.Flag = 0
                return

            if np.isnan(pb).any():
                print('Warning: prob map contains nan')
                print('number of nans'+sum(np.isnan(pb))+' from '+len(pb))


            highres_sum = sum(pb)
            NSIDE = hp.npix2nside(pb.shape[0])
            resolution = (hp.nside2pixarea(NSIDE, degrees=True))

            if reduce_mapresolution == True:

                flag_inf_mu = False
                flag_inf_sigma = False
                if np.isinf(distmu).any():
                    flag_inf_mu = True
                    print('Warning: number of infs in distance array ' +
                          str(sum(np.isinf(distmu)))+' from '+str(len(distmu)))
                    print('prob of correspondend infs region ', sum(
                        pb[np.isinf(distmu)]), ' from ', sum(pb[np.logical_not(np.isinf(distmu))]))

                    pb_hr = pb[np.logical_not(np.isinf(distmu))]

                    distsigma_hr = distsigma[np.logical_not(np.isinf(distmu))]
                    distmu_hr = distmu[np.logical_not(np.isinf(distmu))]
                    distmu_hr_average = np.average(distmu_hr, weights=pb_hr)
                    distmu_std = weighted_avg_and_std(distmu_hr, weights=pb_hr)
                    distmu[np.isinf(distmu)] = 10000  # 10**25
                    #print("number of objects inside the threshold before reducing resolution ",sum(distmu<9000)," of ",len(distmu))
                else:
                    distmu_hr_average = np.average(distmu, weights=pb)
                    distmu_std = weighted_avg_and_std(distmu, weights=pb)

                if np.isinf(distsigma).any():
                    flag_inf_sigma = True
                    print('Warning: number of infs in distance sigma array ' +
                          str(sum(np.isinf(distsigma)))+' from '+str(len(distsigma)))
                    print('prob of correspondend infs region ', sum(
                        pb[np.isinf(distsigma)]), ' from ', sum(pb[np.logical_not(np.isinf(distsigma))]))
                    pb_hr_sigma = pb[np.logical_not(np.isinf(distsigma))]
                    distsigma_hr = distsigma[np.logical_not(np.isinf(distmu))]
                    distsigma_hr_average = np.average(
                        distsigma_hr, weights=pb_hr_sigma)
                    distsigma_std = weighted_avg_and_std(
                        distsigma_hr, weights=pb_hr_sigma)
                    distsigma[np.isinf(distsigma)] = 10000

                else:
                    distsigma_hr_average = np.average(distsigma, weights=pb)
                    distsigma_std = weighted_avg_and_std(distsigma, weights=pb)
                np.save(use_map_info, [
                        distmu_hr_average, distmu_std, distsigma_hr_average, distsigma_std])
                highres_sum = sum(pb)
                NSIDE = hp.npix2nside(pb.shape[0])
                res_high = (hp.nside2pixarea(NSIDE, degrees=True))


                target_res = 2.0
                final_nside_exp = math.ceil(
                    math.log((NSIDE*math.sqrt(res_high))/target_res, 2))  # int(NSIDE/8.0)
                final_nside = 2**final_nside_exp
                final_nside = int(final_nside)

                res_low = hp.nside2pixarea(final_nside, degrees=True)
                res_final = hp.nside2pixarea(final_nside, degrees=True)
                pb = hp.ud_grade(pb, final_nside, power=-2)
                lowres_sum = sum(pb)
                distsigma = hp.ud_grade(distsigma, final_nside)
                distmu = hp.ud_grade(distmu, final_nside)  # power=-2
                print('saving low resolution map')
                hp.write_map(use_map_lowres, m=[pb, distmu, distsigma], nest=False, dtype=None, fits_IDL=True,
                             coord=None, partial=False, column_names=None, column_units=None, extra_header=(), overwrite=False)

                if flag_inf_mu == True:
                    print('prob of reduced correspondend region due to infs ', sum(
                        pb[np.abs(distmu) > (distmu_hr_average+(3*distmu_std))]), ' from ', sum(pb))
                    pb = pb[np.abs(distmu) < (
                        distmu_hr_average+(3*distmu_std))]
                    distsigma = distsigma[np.abs(distmu) < (
                        distmu_hr_average+(3*distmu_std))]
                    distmu = distmu[np.abs(distmu) < (
                        distmu_hr_average+(3*distmu_std))]
                if flag_inf_sigma == True:
                    print('prob of reduced correspondend region due to sigma infs ', sum(pb[np.abs(
                        distsigma) > (distsigma_hr_average+(3*distsigma_std))]), ' from ', sum(pb))
                    pb = pb[np.abs(distsigma) < (
                        distsigma_hr_average+(3*distsigma_std))]
                    distmu = distmu[np.abs(distsigma) < (
                        distsigma_hr_average+(3*distsigma_std))]
                    distsigma = distsigma[np.abs(distsigma) < (
                        distsigma_hr_average+(3*distsigma_std))]  # (distmu_hr_average+(3*distmu_std))

                distmu_lowres_average = np.average(distmu, weights=pb)
                distsigma_lowres_average = np.average(distsigma, weights=pb)

                print("Reducing map resolution ")
                print("Previous resolution and new resolution (deg) ",
                      res_high, " ", res_low)
                print("Sums - high res map prob sum ", highres_sum,
                      " low res map pb sum ", lowres_sum)  # " ",sum(new_pb))
                print("average distance in high and low res map with its standard deviation",
                      distmu_hr_average, "+-", distmu_std, "and ", distmu_lowres_average)
                print("average distance sigma in high and low res map with its standard deviation",
                      distsigma_hr_average, "+-", distsigma_std, "and ", distsigma_lowres_average)
                resolution = res_low
            if read_lowres_map == True:
                print('prob of reduced correspondend region due to infs ', sum(
                    pb[np.abs(distmu) > (distmu_hr_average+(3*distmu_std))]), ' from ', sum(pb))
                pb = pb[np.abs(distmu) < (distmu_hr_average+(3*distmu_std))]
                distsigma = distsigma[np.abs(distmu) < (
                    distmu_hr_average+(3*distmu_std))]
                distmu = distmu[np.abs(distmu) < (
                    distmu_hr_average+(3*distmu_std))]
                print('prob of reduced correspondend region due to sigma infs ', sum(pb[np.abs(
                    distsigma) > (distsigma_hr_average+(3*distsigma_std))]), ' from ', sum(pb))
                pb = pb[np.abs(distsigma) < (
                    distsigma_hr_average+(3*distsigma_std))]
                distmu = distmu[np.abs(distsigma) < (
                    distsigma_hr_average+(3*distsigma_std))]
                distsigma = distsigma[np.abs(distsigma) < (
                    distsigma_hr_average+(3*distsigma_std))]  # (distmu_hr_average+(3*distmu_std))

            if np.isinf(distmu).any():
                print('Warning: number of infs in distance array ' +
                      str(sum(np.isinf(distmu)))+' from '+str(len(distmu)))
                print('prob of correspondend infs region ', sum(
                    pb[np.isinf(distmu)]), ' from ', sum(pb[np.logical_not(np.isinf(distmu))]))

                pb = pb[np.logical_not(np.isinf(distmu))]
                distsigma = distsigma[np.logical_not(np.isinf(distmu))]
                distmu = distmu[np.logical_not(np.isinf(distmu))]

            if np.isnan(distmu).any():
                print('Warning: distance map contains nan')
                print('number of nans '+str(sum(np.isnan(distmu))) +
                      ' from '+str(len(distmu)))
                print('prob of correspondend nan region ', sum(
                    pb[np.isnan(distmu)]), ' from ', sum(pb[np.logical_not(np.isnan(distmu))]))
                pb = pb[np.logical_not(np.isnan(distmu))]
                distsigma = distsigma[np.logical_not(np.isnan(distmu))]
                distmu = distmu[np.logical_not(np.isnan(distmu))]

            idx_sort = np.argsort(pb)
            idx_sort_up = list(reversed(idx_sort))

            sum_ = 0.
            id_c = 0
            sum_full = 0
            id_full = 0
            area_max = sum(pb)
            id_deep = 0

            while (sum_full < min(area_max-0.01, 0.98)) and (id_full < len(idx_sort_up)):
                this_idx = idx_sort_up[id_full]
                sum_full = sum_full+pb[this_idx]
                id_full = id_full+1
            total_area = id_full*resolution
            print("Total event area (deg)="+str(id_full*resolution)+" ")
            while (sum_ < area_covered) and (id_c < len(idx_sort_up)):
                this_idx = idx_sort_up[id_c]
                sum_ = sum_+pb[this_idx]

                if m_exp_kncalc == True:
                    #print ("m_exp=",m_exp)
                    #print ("sum_=",sum_)
                    #print ("id_deep=",id_deep)
                    if sum_ < deep_coverage:
                        id_deep = id_deep+1
                    #print ("id_deep=",id_deep)

                id_c = id_c+1

            idx_sort_full = idx_sort_up[:id_full]

            idx_sort_uncovered = idx_sort_up[id_c:]
            if m_exp_kncalc == True:
                print("number of deep coverage pixels and total covered are pixels=" +
                      str(id_deep)+" "+str(id_c))
            if (id_deep == 0) and (m_exp == True):
                print(
                    "===== Warning: number of deep coverage is 0, switching to single exposure mode")
                m_exp_kncalc = False
                self.sw_mexp = True
            else:
                self.sw_mexp = False
            if m_exp_kncalc == False:
                idx_sort_cut = idx_sort_up[:id_c]

            else:
                idx_sort_cut = idx_sort_up[id_deep:id_c]
                idx_sort_deep = idx_sort_up[:id_deep]
                pb_covered_deep = pb[idx_sort_deep]
                num_pix_covered_deep = len(pb_covered_deep)
                distmu_covered_deep = distmu[idx_sort_deep]
                distsigma_covered_deep = distsigma[idx_sort_deep]

            pb_covered = pb[idx_sort_cut]
            num_pix_covered = len(pb_covered)
            distmu_covered = distmu[idx_sort_cut]
            distsigma_covered = distsigma[idx_sort_cut]

            # FIXME

            try:
                distmu_average = np.average(distmu_covered, weights=pb_covered)
                distsigma_average = np.average(
                    distsigma_covered, weights=pb_covered)
            except:
                self.Flag = 0
                return
            print("the distance average="+str(distmu_average))
            print("the distance sigma="+str(distsigma_average))
            if m_exp_kncalc == False:
                print("the area covered="+str(area_covered))
                print("the prob covered="+str(np.sum(pb_covered)))

            distmu_full = distmu[idx_sort_full]
            pb_full = pb[idx_sort_full]
            distsigma_full = distsigma[idx_sort_full]

            if (distmu_full < 0.0).any():
                print('Warning: prob of distmu <0.0 in the full region region ', sum(
                    pb_full[distmu_full < 0.0]), ' from ', sum(pb_full[distmu_full > 0.0]))
                distmu_full[distmu_full < 0.0] = 0.01
            if (distmu_covered < 0.0).any():
                print('Warning: prob of distmu<0.0 region ', sum(
                    pb_covered[distmu_covered < 0.0]), ' from ', sum(pb_covered[distmu_covered > 0.0]))
                distmu_covered[distmu_covered < 0.0] = 0.01
            if m_exp_kncalc == True:
                if (distmu_covered < 0.0).any():
                    print('Warning: prob of distmu<0.0 in the deep region ', sum(
                        pb_covered_deep[distsigma_covered_deep < 0.0]), ' from ', sum(pb_covered_deep[distsigma_covered_deep > 0.0]))
                    distsigma_covered_deep[distsigma_covered_deep < 0.0] = 0.01

            pb_vol_norm = np.sum(np.multiply(distmu_full, pb_full))

            pb_vol_covered_all = np.sum(
                np.multiply(distmu_covered, pb_covered))

            pb_vol_covered_all = pb_vol_covered_all/pb_vol_norm

            print("the prob volume covered (except deep region if any)=" +
                  str(pb_vol_covered_all))

            if np.isnan(pb_full).any():
                print('Warning: prob map full cut contains nan')
                print('number of nans'+sum(np.isnan(pb_full)) +
                      ' from '+len(pb_full))

            for k in range(0, len(pb_covered)):
                weights_pix = [norm.pdf(x.value, loc=float(distmu_covered[k]), scale=float(
                    distsigma_covered[k])) for x in cosmo.luminosity_distance(template_df_full['ZMEAN'].values)]
                weights_pix_norm = [norm.pdf(x.value, loc=float(distmu_covered[k]), scale=float(
                    distsigma_covered[k])) for x in cosmo.luminosity_distance(np.unique(template_df_full['ZMEAN'].values))]


                pb_vol_covered = (
                    pb_covered[k] * distmu_covered[k])/pb_vol_norm
                if np.sum(weights_pix_norm) == 0.0:
                    print("the weights pix sum is 0, skipping pixel")
                    continue
                weights_pix = (weights_pix / np.sum(weights_pix_norm))
                weights_pix = weights_pix*float(pb_vol_covered)
                if k == 0:
                    weights_pix_area = weights_pix
                else:
                    weights_pix_area = np.add(weights_pix_area, weights_pix)

            print("the weights sum="+str(np.sum(weights_pix_area)))
            template_df_full['WEIGHT'] = weights_pix_area
            if m_exp_kncalc == True:
                print("Getting the weights sum of Deep region with " +
                      str(len(pb_covered_deep))+" voxels")
                for k in range(0, len(pb_covered_deep)):
                    weights_pix_deep = [norm.pdf(x.value, loc=float(distmu_covered_deep[k]), scale=float(
                        distsigma_covered_deep[k])) for x in cosmo.luminosity_distance(template_df_full['ZMEAN'].values)]
                    weights_pix_norm_deep = [norm.pdf(x.value, loc=float(distmu_covered_deep[k]), scale=float(
                        distsigma_covered_deep[k])) for x in cosmo.luminosity_distance(np.unique(template_df_full['ZMEAN'].values))]

                    pb_vol_covered_deep = (
                        pb_covered_deep[k] * distmu_covered_deep[k])/pb_vol_norm
                    if np.sum(weights_pix_norm) == 0.0:
                        print("the weights pix sum is 0, skipping pixel")
                        continue
                    weights_pix_deep = (
                        weights_pix_deep / np.sum(weights_pix_norm_deep))
                    weights_pix_deep = weights_pix_deep * \
                        float(pb_vol_covered_deep)
                    if k == 0:
                        weights_pix_area_deep = weights_pix_deep
                    else:
                        weights_pix_area_deep = np.add(
                            weights_pix_area_deep, weights_pix_deep)

                print("the weights sum of Deep area=" +
                      str(np.sum(weights_pix_area_deep)))
                template_df_full['WEIGHT_deep'] = weights_pix_area_deep
                np.save(use_map_weights, [
                        weights_pix_area, weights_pix_area_deep])
                np.save(use_map_weights_info, [
                        num_pix_covered*resolution, num_pix_covered_deep*resolution, resolution])
            else:
                np.save(use_map_weights, weights_pix_area)
                np.save(use_map_weights_info, [
                        num_pix_covered*resolution, resolution])

            self.Resolution = resolution
            self.area_deg = num_pix_covered*resolution
            if m_exp_kncalc == True:
                self.area_deg_deep = num_pix_covered_deep*resolution

        elif preprocessed_weights == True:

            template_df_full['WEIGHT'] = weights_pre
            if m_exp_kncalc == True:
                template_df_full['WEIGHT_deep'] = weights_pre_deep

        else:
            weights = [norm.pdf(x.value, loc=float(self.distance), scale=float(
                self.distance_err)) for x in cosmo.luminosity_distance(template_df_full['ZMEAN'].values)]
            weights_norm = [norm.pdf(x.value, loc=float(self.distance), scale=float(
                self.distance_err)) for x in cosmo.luminosity_distance(np.unique(template_df_full['ZMEAN'].values))]

            template_df_full['WEIGHT'] = (
                weights / np.sum(weights_norm))  # *area_covered
            print("the weights sum (no map)=" +
                  str(np.sum(template_df_full['WEIGHT'])))
            print("probability covered (no map)="+str(area_covered))

        template_df_later['WEIGHT'] = template_df_full['WEIGHT'].values
        if m_exp_kncalc == True:
            template_df_later['WEIGHT_deep'] = template_df_full['WEIGHT_deep'].values

        self.delta_mjd_later = delta_mjd_later
        self.template_df_later = template_df_later
        self.template_df_full = template_df_full
        self.Flag = 1
        self.mult_exp = m_exp_kncalc
        return
