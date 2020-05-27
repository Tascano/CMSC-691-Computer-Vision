#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dc_image.h"


#define CANNY_THRESH 25
#define CANNY_BLUR   5



#define MIN(a,b)  ( (a) < (b) ? (a) : (b) )
#define MAX(a,b)  ( (a) > (b) ? (a) : (b) )
#define ABS(x)    ( (x) <= 0 ? 0-(x) : (x) )

#define KM_NUM_CLUSTERS   10
#define KM_NUM_ITER       100

#define KM_RED_SCALE   5.0
#define KM_GREEN_SCALE 5.0
#define KM_BLUE_SCALE  5.0
#define KM_X_SCALE     1.0
#define KM_Y_SCALE     1.0

#define NUM_WHEEL 20
byte wheel_R[] = {255, 255,   0,   0,   0, 255, 128, 128,   0,   0,   0, 128, 255, 255, 128, 128, 128, 255, 255,   0};
byte wheel_G[] = {  0, 255, 255, 255,   0,   0,   0, 128, 128, 128,   0,   0, 128, 255, 255, 255, 128, 128, 255,   0};
byte wheel_B[] = {  0,   0,   0, 255, 255, 255,   0,   0,   0, 128, 128, 128, 128, 128, 128, 255, 255, 255, 255,   0};

typedef struct Point {
	double r,g,b,x,y;    // The position of the point
} Point;

float Dist(Point a, Point b)
{
	// Different between two points
	float dr = (a.r-b.r) * KM_RED_SCALE;    // Mahalobis Distance
	float dg = (a.g-b.g) * KM_GREEN_SCALE;
	float db = (a.b-b.b) * KM_BLUE_SCALE;
	float dx = (a.x-b.x) * KM_X_SCALE;
	float dy = (a.y-b.y) * KM_Y_SCALE;
	
	// Pythagorean Theorem
//	return sqrt(dr*dr + dg*dg + db*db + dx*dx + dy*dy);

	// Manhattan Distance
	return ABS(dr) + ABS(dg) + ABS(db) + ABS(dx) + ABS(dy);
	
	// Min Manhattan Distance
//	return MIN(MIN(MIN(MIN(ABS(dr),ABS(dg)),ABS(db)),ABS(dx)),ABS(dy));
}

// Rand sometimes does not return a large enough number
unsigned int BigRand()
{
	return ( ((rand()&0xffff)<<16) | (rand()&0xffff) );
}


int main()
{
	int y,x,i,j,iter;
	int rows, cols, chan;

	//-----------------
	// Read the image
	//-----------------
	byte ***img = LoadRgb("tiger.jpg", &rows, &cols, &chan);
	printf("img %p rows %d cols %d chan %d\n", img, rows, cols, chan);
	
	SaveRgbPng(img, "out/1_img.png", rows, cols);


	//----------------
	// Make a visualiztion image fo the result
	//----------------

	byte ***result = malloc3d(rows,cols,3);

	// Each cluster is a different color
	int *cluster_red   = (int*)malloc(KM_NUM_CLUSTERS * sizeof(int));
	int *cluster_green = (int*)malloc(KM_NUM_CLUSTERS * sizeof(int));
	int *cluster_blue  = (int*)malloc(KM_NUM_CLUSTERS * sizeof(int));
	for (j=0; j<KM_NUM_CLUSTERS; j++)
	{
		cluster_red[j]   = rand() % 256;
		cluster_green[j] = rand() % 256;
		cluster_blue[j]  = rand() % 256;
	}

	//-----------------
	// Create a set of points
	//-----------------
	
	int nPoints = rows * cols;
	Point *points       = (Point*)malloc(nPoints * sizeof(Point));    // Array of points
	int   *pointCluster = (int*)malloc(nPoints * sizeof(int));		// Which cluster is the

	// Copy points to an array and scale them
	i=0;
	for (y=0; y<rows; y++) {
		for (x=0; x<cols; x++) {
			points[i].r = img[y][x][0];
			points[i].g = img[y][x][1];
			points[i].b = img[y][x][2];
			points[i].x = x;
			points[i].y = y;
			i++;
		}
	}
	
	//----------------
	// Make a set of cluster centers
	//----------------	
	Point *clusters      = malloc(KM_NUM_CLUSTERS * sizeof(Point));  // Array of clustes
	Point *clusterTotal  = malloc(KM_NUM_CLUSTERS * sizeof(Point));
	int   *clusterCount  = malloc(KM_NUM_CLUSTERS * sizeof(int));    // How many points in the cluster ?
	
	// Assign to random points
	for (j=0; j<KM_NUM_CLUSTERS; j++) {
		clusters[j] = points[ BigRand() % nPoints ];   // Cluster is random point

		printf("Cluster %d\n", j);
		printf("clusters[%d]: r %.3f g %.3f b %.3f  x %.3f y %.3f\n", j,
			clusters[j].r, clusters[j].g, clusters[j].b,
			clusters[j].x, clusters[j].y);
	}

	//----------------
	// Run K-means
	//----------------
	for (iter = 0; iter < KM_NUM_ITER; iter++)
	{
		printf("-----------------------------------------\n");
		printf(" KMEANS iter: %d\n", iter);
		
		//----
		// Assign to the closest cluster center
		//----
		for (i=0; i<nPoints; i++)
		{
			// Find the closes center
			int   closest_clust = 0;
			float closest_dist  = 9999999.0;
			for (j=0; j<KM_NUM_CLUSTERS; j++)
			{
				float dist = Dist(points[i], clusters[j]);
				if (dist < closest_dist)
				{
					closest_dist  = dist;
					closest_clust = j;
				}
			}
			
			// Assign the closest center
			pointCluster[i] = closest_clust;
		}
		
		//----
		// Take the mean of each cluster center
		//----
		
		// Reset the clusters to zero
		for (j=0; j<KM_NUM_CLUSTERS; j++)
		{
			//printf("Cluster %d\n", j);
			printf("clusters[%d]: r %.3f g %.3f b %.3f  x %.3f y %.3f\t\tcount %d\n", j,
				clusters[j].r, clusters[j].g, clusters[j].b,
				clusters[j].x, clusters[j].y, clusterCount[j]);

			clusterTotal[j].r = 0.0;   // Total is zero
			clusterTotal[j].g = 0.0;
			clusterTotal[j].b = 0.0;
			clusterTotal[j].x = 0.0;
			clusterTotal[j].y = 0.0;
			clusterCount[j] = 0;   // Count is zero
		}
		
		// Calculate the total and count
		for (i=0; i<nPoints; i++)
		{
			j = pointCluster[i];  // Which cluster

			// Increase total
			clusterTotal[j].r += points[i].r;
			clusterTotal[j].g += points[i].g;
			clusterTotal[j].b += points[i].b;
			clusterTotal[j].x += points[i].x;
			clusterTotal[j].y += points[i].y;
			
			// Increase count
			clusterCount[j]++;
		}

		// Mean is total divided by count
		for (j=0; j<KM_NUM_CLUSTERS; j++)
		{
			clusters[j].r = clusterTotal[j].r / clusterCount[j];
			clusters[j].g = clusterTotal[j].g / clusterCount[j];
			clusters[j].b = clusterTotal[j].b / clusterCount[j];
			clusters[j].x = clusterTotal[j].x / clusterCount[j];
			clusters[j].y = clusterTotal[j].y / clusterCount[j];
		}
		
		
		// Visualize the clusters result
		i=0;
		for (y=0; y<rows; y++)
		{
			for (x=0; x<cols; x++)
			{
				j = pointCluster[i];            // Which cluster ?
				result[y][x][0] = wheel_R[j % NUM_WHEEL];   // Set out pixel to cluster color
				result[y][x][1] = wheel_G[j % NUM_WHEEL];   // Set out pixel to cluster color
				result[y][x][2] = wheel_B[j % NUM_WHEEL];   // Set out pixel to cluster color		

				i++;   // next point
			}
		}

		// Write each cluster to a separate file
		char outpath[4096];
		sprintf(outpath, "out/2_result_iter_%05d.png", iter);
		SaveRgbPng(result, outpath, rows, cols);
		
//		printf("press enter to continue\n");
//		fgetc(stdin);
	}

	

	printf("Done!\n");

	return 0;
}

/*

	printf("load image\n");
	byte *data = stbi_load("puppy.jpg", &cols, &rows, &chan, 4);

	printf("data = %p\n", data);
	int rt=stbi_write_png("output.png", cols, rows, 4, data, cols*4);
*/