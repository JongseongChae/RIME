from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from algorithm.arguments import get_args

if __name__ == '__main__':
    start_time = datetime.now()
    args = get_args()

    if args.sampled_envs == 2:
        # --------------------------------
        sampled_name = args.env_name.split("-")
        sampled_per = args.env_parameter.lower()[0]
        sampled_degree = ["050", "150"]

        args.expert_path1 = "ED_" + sampled_name[0] + sampled_degree[0] + sampled_per + '-' + sampled_name[1]
        args.expert_path2 = "ED_" + sampled_name[0] + sampled_degree[1] + sampled_per + '-' + sampled_name[1]

        args.num_env_steps = 10000000
        # --------------------------------
        from train_n_evalauate import learn_the_2_sampled_envs as main
    elif args.sampled_envs == 3:
        # --------------------------------
        sampled_name = args.env_name.split("-")
        sampled_per = args.env_parameter.lower()[0]
        sampled_degree = ["050", "150"]

        args.expert_path1 = "ED_" + sampled_name[0] + sampled_degree[0] + sampled_per + '-' + sampled_name[1]
        args.expert_path2 = "ED_" + sampled_name[0] + '-' + sampled_name[1]  # nominal environment
        args.expert_path3 = "ED_" + sampled_name[0] + sampled_degree[1] + sampled_per + '-' + sampled_name[1]

        args.num_env_steps = 10000000
        # --------------------------------
        from train_n_evalauate import learn_the_3_sampled_envs as main
    elif args.sampled_envs == 4:
        # --------------------------------
        sampled_name = args.env_name.split("-")
        sampled_degree = ["050g050m", "150g050m", "050g150m", "150g150m"]

        args.expert_path1 = "ED_" + sampled_name[0] + sampled_degree[0] + '-' + sampled_name[1]
        args.expert_path2 = "ED_" + sampled_name[0] + sampled_degree[1] + '-' + sampled_name[1]
        args.expert_path3 = "ED_" + sampled_name[0] + sampled_degree[2] + '-' + sampled_name[1]
        args.expert_path4 = "ED_" + sampled_name[0] + sampled_degree[3] + '-' + sampled_name[1]

        args.num_env_steps = 5000000
        # --------------------------------
        from train_n_evalauate import learn_the_4_sampled_envs as main
    else:
        print("Check sampled-envs")

    seeds = [10010, 10020, 10030, 10040, 10050, 10060, 10070, 10080, 10090, 10100]

    args_pool = [0]*len(seeds)
    data = []
    record_perf = []

    for i in range(len(seeds)):
        args_pool[i] = deepcopy(args)
        args_pool[i].seed = seeds[i]

    with Pool(len(seeds)) as p:
        data = p.map(main, args_pool)

    if (args.sampled_envs == 2) or (args.sampled_envs == 3):
        def classify_data(seeds, data, store):
            for i in range(len(seeds)):
                store.append(data[i][0])

        def expert_gail(env_name, env_para):
            real_env_name = env_name.split("_")[1]
            env_pertur = env_name.split("_")[1][-7:-4]

            if "Hopper" in real_env_name:
                limku = 5000
                limkl = -200
                if "gravity" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Hopper050g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Hopper150g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_Hopper-v2_gravity.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
                elif "mass" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Hopper050m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Hopper150m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_Hopper-v2_mass.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
            elif "Walker" in real_env_name:
                limku = 6100
                limkl = -300
                if "gravity" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Walker2d050g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Walker2d150g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_Walker2d-v2_gravity.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
                elif "mass" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Walker2d050m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Walker2d150m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_Walker2d-v2_mass.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
            elif "HalfCheetah" in real_env_name:
                limku = 6100
                limkl = -1800
                if "gravity" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_HalfCheetah050g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_HalfCheetah150g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_HalfCheetah-v2_gravity.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
                elif "mass" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_HalfCheetah050m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_HalfCheetah150m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_HalfCheetah-v2_mass.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
            elif "Ant" in real_env_name:
                limku = 5200
                limkl = -200
                if "gravity" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Ant050g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Ant150g-v2_gravity.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_Ant-v2_gravity.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
                elif "mass" in env_para:
                    if "050" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Ant050m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    elif "150" in env_pertur:
                        data = np.load("./results/gail_performance/gail_gp_Ant150m-v2_mass.npz")["record_perf"][0]
                        result = str(env_pertur) + str(env_para[0])
                    else:
                        data = np.load("./results/gail_performance/gail_gp_Ant-v2_mass.npz")["record_perf"][0]
                        result = str(100) + str(env_para[0])
            return data, result, limku, limkl

        classify_data(seeds, data, record_perf)

        avg_record_perf = np.mean(np.concatenate(record_perf, axis=1), axis=1)
        std_record_perf = np.std(np.concatenate(record_perf, axis=1), axis=1)

        rec_save_path = 'results/' + str(data[0][-1])
        np.savez(rec_save_path, record_perf=[avg_record_perf, std_record_perf], allow_pickle=True)

        gail1, par1, limku, limkl = expert_gail(args.expert_path1, args.env_parameter)
        if args.sampled_envs == 2:
            gail2, par2, limku2, limkl2 = expert_gail(args.expert_path2, args.env_parameter)
        elif args.sampled_envs == 3:
            gail2, par2, limku2, limkl2 = expert_gail(args.expert_path2, args.env_parameter)
            gail3, par3, limku3, limkl3 = expert_gail(args.expert_path3, args.env_parameter)

        a_t = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
               1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85,
               1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3]

        plt.figure()
        plt.plot(a_t, avg_record_perf, label=args.algo_name)
        plt.fill_between(a_t, avg_record_perf - std_record_perf, avg_record_perf + std_record_perf, alpha=0.1)
        plt.plot(a_t, gail1, label='GAIL+GP-' + par1, alpha=0.4)
        if args.sampled_envs == 2:
            plt.plot(a_t, gail2, label='GAIL+GP-' + par2, alpha=0.4)
        elif args.sampled_envs == 3:
            plt.plot(a_t, gail2, label='GAIL+GP-' + par2, alpha=0.4)
            plt.plot(a_t, gail3, label='GAIL+GP-' + par3, alpha=0.4)
        plt.legend()
        plt.savefig(rec_save_path + '_perf.png')
    elif args.sampled_envs == 4:
        def classify_data(seeds, data, index, store):
            for i in range(len(seeds)):
                store.append(data[i][index])

        classify_data(seeds, data, 0, record_perf)

        avg_record_perf_2par_gm = np.mean(np.concatenate(record_perf, axis=1), axis=1)
        std_record_perf_2par_gm = np.std(np.concatenate(record_perf, axis=1), axis=1)
        perf_2par_gm = np.reshape(avg_record_perf_2par_gm, (6, 6))

        rec_save_path = 'results/' + str(data[0][-1])  # data[2] is excute_name

        np.savez(rec_save_path, record_perf_2par_gm=[avg_record_perf_2par_gm, std_record_perf_2par_gm], allow_pickle=True)

        envwith2par_list = [50, 70, 90, 110, 130, 150]

        plt.figure()
        plt.imshow(perf_2par_gm, cmap='viridis')
        plt.xticks(range(len(envwith2par_list)), envwith2par_list)
        plt.yticks(range(len(envwith2par_list)), envwith2par_list[::-1])
        plt.xlabel('The Percentage of Gravity Perturbation')
        plt.ylabel('The Percentage of Mass Perturbation')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(rec_save_path + '_perf.png')

    print('All process done.')
    print('Time Consumed: {}'.format(datetime.now() - start_time))

    import sys; sys.exit()

