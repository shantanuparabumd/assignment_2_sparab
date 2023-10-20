if (step % args.vis_freq) == 0:
            # visualization block
            #  rend = 
            print("Rendering Ground Truth")
            generate_gif_from_mesh(mesh_gt,'images/mesh_gt.gif')

            if args.type=='vox':
                print("Rendering Voxel Predictions")
                predictions=predictions.sqeeze(0)
                generate_gif_from_voxels(predictions,'images/vox_prediction.gif')

            elif args.type=='point':
                print("Rendering Point Predictions")
                print(predictions.shape)

                generate_gif_from_point(predictions,'images/point_prediction.gif')

            elif args.type=='mesh':
                print("Rendering Mesh Predictions")
                generate_gif_from_mesh(predictions,'images/mesh_prediction.gif')