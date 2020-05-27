#include "./utils/image_tools.h"
#include "./utils/ransac.h"

#define NUM_IMG 4
char *get_img_name(int ind)
{
	if (ind == 0)
	{
		return "pentagon.png";
	}
	if (ind == 1)
	{
		return "sidewalk.png";
	}
	if (ind == 2)
	{
		return "puppy.jpg";
	}
	if (ind == 3)
	{
		return "building.png";
	}
	return NULL;
}

int main()
{
	for (int ind_img = 0; ind_img < NUM_IMG; ind_img++)
	{
		char *img_path = get_img_name(ind_img);
		Mat8U *edges = canny(img_path);

		if (edges == NULL)
			continue;

		Mat8U *edgesTemp = copy_mat_8u(edges);
		Mat8U *colorImg = load_image(img_path);
		Mat8U *outputImg = clone_mat_8u(colorImg);
		int width = edges->width;
		int height = edges->height;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				outputImg->data[3 * (y * width + x) + 2] = edges->data[y * width + x]; // red
				outputImg->data[3 * (y * width + x) + 1] = edges->data[y * width + x]; // green
				outputImg->data[3 * (y * width + x) + 0] = edges->data[y * width + x]; // blue
			}
		}

		double Threshold = 1;
		double Confidence = 0.99;
		double ApproximateInlierFraction = 0.4;

		int count = 0;

		int min_inlier = 100;
		int prominent = 0;

		Vector max_inlier;
		vec_init(&max_inlier);

		int stop_cond = 0;
		while (true)
		{
			TPoint Point0;
			TPoint Point1;

			int num_points = 0;
			TPoint *point_idxs = get_points(edgesTemp, &num_points);
			if (point_idxs == NULL)
				break;
			int numData = num_points;
			int Inliers[num_points];

			ransac_fitting(point_idxs, num_points, Threshold, Confidence, ApproximateInlierFraction, &Point0, &Point1, Inliers);
			int count = 0;
			for (size_t Index = 0; Index != numData; ++Index)
			{
				if (Inliers[Index])
				{
					count++;
				}
			}


				prominent++;
				stop_cond++;

				float pt_y_max = 0;
				float pt_y_min = FLT_MAX;
				float y_max_idx = 0;
				float y_min_idx = 0;
				for (size_t Index = 0; Index != numData; ++Index)
				{

					if (Inliers[Index])
					{
						if (point_idxs)
						{
							int idx = point_idxs[Index].x + point_idxs[Index].y * edges->width;
							if (pt_y_max < point_idxs[Index].y)
							{
								pt_y_max = point_idxs[Index].y;
								y_max_idx = Index;
							}
							if (pt_y_min > point_idxs[Index].y)
							{
								pt_y_min = point_idxs[Index].y;
								y_min_idx = Index;
							}
							edgesTemp->data[idx] = 0;
						}
					}
				}
				pt2i fstPoint = {Point0.x, Point0.y};
				pt2i sndPoint = {Point1.x, Point1.y};
				uchar color[3] = {0, 255, 0};
				Mat8U *newImg = draw_line(outputImg, fstPoint, sndPoint, color);
				outputImg = copy_mat_8u(newImg);
				free(newImg);


			if (point_idxs)
				free(point_idxs);
			if (prominent > 100 || stop_cond > 200 || count == 0)
				break;
		}

		char out_filename[255];
		sprintf(out_filename, "output/out_%s", img_path);
		save_rgb_image(outputImg, out_filename);
		free(edges);
		free(edgesTemp);
		free(outputImg);
		free(colorImg);
	}

	return 0;
}
