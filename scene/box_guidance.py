import torch


class BoxGuidance:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.all_boundary_indices = []
        self.box_minn_tensor = None
        self.box_maxx_tensor = None
        self.box_scale = None
        self.lengths_tensor = None

    # def set_gaussians(self, gaussians):
    #     self.gaussians = gaussians




    def save_to_csv(self):
        ############## CSV save ##############
        import csv

        # Define the CSV file name
        csv_file_name = 'self.all_boundary_indices.csv'

        # Write the list of indices to the CSV file
        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for indices in self.all_boundary_indices:
                # Check if indices is a tensor with just one element
                if torch.numel(indices) == 1:
                    # Write a single element tensor as a one-element list
                    writer.writerow([indices.item()])
                else:
                    # Convert tensor to list and write
                    writer.writerow(indices.tolist())

    def get_box_boundary(self, iteration):
        ################## box boundary #########################

        pcd = self.gaussians.tensor_to_pcd(self.gaussians.get_xyz)
        box_data = self.gaussians.get_boxes(pcd)
        self.box_minn_tensor = box_data[:, :3]  # All rows, first 3 columns
        self.box_maxx_tensor = box_data[:, 3:6]  # All rows, columns 3 to 5
        self.box_scale = box_data[:, 6:]  # All rows, last column

        def points_within_boundaries():
            # Move data to GPU
            points = self.gaussians.get_xyz.cuda()
            min_boundaries = self.box_minn_tensor.cuda()
            max_boundaries = self.box_maxx_tensor.cuda()

            # Expand dimensions for broadcasting
            points_expanded = points.unsqueeze(1)  # Shape: [num_points, 1, 3]

            # Check if points are within boundaries
            within_min = points_expanded >= min_boundaries  # Shape: [num_points, num_boundaries, 3]
            within_max = points_expanded <= max_boundaries  # Shape: [num_points, num_boundaries, 3]

            # Both conditions must be true for all coordinates
            within_boundaries = torch.all(within_min & within_max,
                                          dim=2)  # Shape: [num_points, num_boundaries]

            return within_boundaries

        # Checking which points are within which boundaries
        _points_within_boundaries = points_within_boundaries()

        # Convert the boolean tensor to an integer tensor
        points_within_boundaries_int = _points_within_boundaries.int()

        # Now apply argmax
        first_boundary_indices = torch.argmax(points_within_boundaries_int, dim=1)

        # For finding all boundary indices for each point
        self.all_boundary_indices = [torch.nonzero(_points_within_boundaries[i]).squeeze() for i in
                                range(_points_within_boundaries.shape[0])]

        # return self.all_boundary_indices
        ######################################################
        # self.all_boundary_indices = get_box_boundary()


    def prune_points_most_boxes(self, iteration):
        ############## points pruning with most boxes ##############
        # Step 1: Mask for elements with only {bounding_box_num} boundary
        bounding_box_num = max(
            (sub_tensor.size(0) for sub_tensor in self.all_boundary_indices if sub_tensor.dim() > 0), default=0)
        # bounding_box_num = 0
        one_boundary_mask = torch.tensor(
            [indices.numel() == bounding_box_num for indices in self.all_boundary_indices])

        print(f"\n[iteration {iteration}] Number of Gaussians before pruning: {self.gaussians.get_xyz.shape[0]}")
        self.gaussians.prune_points(one_boundary_mask)
        print(f"[iteration {iteration}] Number of Gaussians after pruning: {self.gaussians.get_xyz.shape[0]}")
        ############################################################
        # prune_points_most_boxes(self.all_boundary_indices)


    def box_indices_to_csv(self):
        ############## CSV save ##############
        import csv

        # Define the CSV file name
        csv_file_name = 'all_boundary_indices.csv'

        # Write the list of indices to the CSV file
        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for indices in self.all_boundary_indices:
                # Check if indices is a tensor with just one element
                if torch.numel(indices) == 1:
                    # Write a single element tensor as a one-element list
                    writer.writerow([indices.item()])
                else:
                    # Convert tensor to list and write
                    writer.writerow(indices.tolist())
        #######################################
        # box_indices_to_csv(self.all_boundary_indices)

    def mask_box(self):
        ############## box mask ##############
        # Assuming self.all_boundary_indices is a list of tensors
        # Reshape each tensor in the list to be 1D if it's not already
        reshaped_all_boundary_indices = [indices.reshape(-1) for indices in self.all_boundary_indices]

        # Now concatenate the reshaped tensors
        flat_indices = torch.cat(reshaped_all_boundary_indices)

        # Create a mask where each element is True if it's equal to 3
        mask = (flat_indices == 3)

        # Count the number of True values in the mask
        count_boundary_index_3 = torch.sum(mask).item()

        # Initialize a variable to store the minimum length
        min_length = float('inf')  # Start with infinity

        # Iterate through all tensors in self.all_boundary_indices
        for indices in self.all_boundary_indices:
            # Update min_length if the current tensor is smaller
            min_length = min(min_length, indices.numel())

        # Create a list of the lengths of each tensor in self.all_boundary_indices
        lengths = [indices.numel() for indices in self.all_boundary_indices]

        # Convert the list of lengths to a PyTorch tensor
        lengths_tensor = torch.tensor(lengths).float().cuda()  # Ensure it's a floating point tensor

        self.lengths_tensor = lengths_tensor
        #####################################

    def prune_opacity(self):
        ############## opacity ##############
        lengths_tensor = self.lengths_tensor

        # Normalize to [0, 1]
        min_val = torch.min(lengths_tensor).cuda()
        max_val = torch.max(lengths_tensor).cuda()
        lengths_norm_tensor = (lengths_tensor - min_val) / (max_val - min_val)

        # Shift to [-1, 1]
        lengths_norm_tensor = (lengths_norm_tensor * 2 - 1) * 0.1

        self.gaussians.set_opacity(self.gaussians.get_opacity + lengths_norm_tensor.unsqueeze(0).T)
        #####################################

    def prune_opacity_beta(self):
        ########## opacity - Beta ##########
        from torch.distributions.beta import Beta
        lengths_tensor = self.lengths_tensor
        # Assuming lengths_tensor is already defined and is a tensor
        min_val = torch.min(lengths_tensor).cuda()
        max_val = torch.max(lengths_tensor).cuda()
        lengths_norm_tensor = (lengths_tensor - min_val) / (max_val - min_val)

        # Define alpha and beta parameters for the beta distribution
        alpha, beta = 0.5, 0.5  # Example values, change them according to your needs
        beta_distribution = Beta(alpha, beta)

        # Sample from the beta distribution
        beta_samples = beta_distribution.sample(lengths_norm_tensor.shape).cuda()

        # Now scale these samples to [-1, 1] to match the scaling of lengths_norm_tensor
        beta_samples_scaled = (beta_samples * 2 - 1) * 0.1

        print("beta")

        # Modify the opacity
        self.gaussians.set_opacity(self.gaussians.get_opacity + beta_samples_scaled.unsqueeze(0).T)
        #####################################

    def prune_points_least_boxes(self, iteration):
        ############## prune ##############
        box_scale = self.box_scale.cuda()
        min_val = torch.min(box_scale).cuda()
        max_val = torch.max(box_scale).cuda()

        box_scale_norm_tensor = (box_scale - min_val) / (max_val - min_val)

        mean = torch.mean(box_scale_norm_tensor)
        sigma = torch.std(box_scale_norm_tensor)

        # Create a boolean mask where the condition is true
        condition = box_scale_norm_tensor > mean + 2 * sigma

        # Get indices of elements satisfying the condition
        selected_indices = torch.nonzero(condition, as_tuple=False)[:, 0]

        # Step 1: Mask for elements with only one boundary
        bounding_box_num = 1
        one_boundary_mask = torch.tensor([indices.numel() <= bounding_box_num for indices in self.all_boundary_indices])

        # Step 2: Create a mask for each point in self.all_boundary_indices
        selected_indices_mask_per_point = [torch.any(torch.isin(indices, selected_indices)) for indices in
                                           self.all_boundary_indices]

        # Convert the list of booleans to a tensor if needed
        selected_indices_mask_per_point_tensor = torch.tensor(selected_indices_mask_per_point)

        # Step 3: Combine the masks
        # Since flat_self.all_boundary_indices is flat, we need to reshape selected_indices_mask to match the shape of one_boundary_mask
        selected_indices_mask_reshaped = selected_indices_mask_per_point_tensor.view(len(self.all_boundary_indices), -1).any(
            dim=1)
        final_mask = one_boundary_mask & selected_indices_mask_reshaped
        print(f"\n[iteration {iteration}] Number of Gaussians before pruning: {self.gaussians.get_xyz.shape[0]}")
        self.gaussians.prune_points(final_mask)
        print(f"[iteration {iteration}] Number of Gaussians after pruning: {self.gaussians.get_xyz.shape[0]}")



