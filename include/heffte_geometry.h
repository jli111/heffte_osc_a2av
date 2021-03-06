/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_GEOMETRY_H
#define HEFFTE_GEOMETRY_H

#include <array>
#include <valarray>
#include "heffte_utils.h"

namespace heffte {

/*!
 * \ingroup fft3d
 * \addtogroup fft3dgeometry Box-geometry operations
 *
 * HeFFTe operates with indexes that are distributed in boxes across the mpi ranks.
 * Several methods help manipulate such vectors of boxes,
 * note that in each instance the order of the boxes in a single vector should always match.
 */

/*!
 * \ingroup fft3d
 * \brief A generic container that describes a 3d box of indexes.
 *
 * The box is defined by three low and three high indexes and holds all indexes from low to high
 * including the low and high. For example,
 * \code
 *  box3d box({0, 0, 0}, {0, 1, 2});
 *  // box will hold the 6 indexes
 *  // (0, 0, 0), (0, 0, 1), (0, 0, 2)
 *  // (0, 1, 0), (0, 1, 1), (0, 1, 2)
 * \endcode
 *
 * The box3d also defines a size field that holds the number of indexes in each direction,
 * for the example above size will be {1, 2, 3}.
 *
 * In addition to the low and high indexes, the box has orientation with respect to the i j k indexing,
 * indicating which are the fast, middle and slow growing dimensions.
 * For example, suppose we want to access the entries of a \b box stored in a \b data array
 * \code
 *  // if box.order is {0, 1, 2}
 *  int const plane = box.size[0] * box.size[1];
 *  int const lane  = box.size[0];
 *  for(int i = box.low[0]; i <= box.high[0]; i++)
 *      for(int j = box.low[1]; j <= box.high[1]; j++)
 *          for(int k = box.low[2]; j <= box.high[2]; j++)
 *              std::cout << data[k * plane + j * line + i] << "\n";
 * \endcode
 * However,
 * \code
 *  // if box.order is {2, 0, 1}
 *  int const plane = box.size[box.order[0]] * box.size[box.order[1]];
 *  int const lane  = box.size[box.order[0]];
 *  for(int i = box.low[0]; i <= box.high[0]; i++)
 *      for(int j = box.low[1]; j <= box.high[1]; j++)
 *          for(int k = box.low[2]; j <= box.high[2]; j++)
 *              std::cout << data[j * plane + i * line + k] << "\n";
 * \endcode
 *
 * The order must always hold a permutation of the entries 0, 1, and 2.
 */
struct box3d{
    //! \brief Constructs a box from the low and high indexes, the span in each direction includes the low and high (uses default order).
    box3d(std::array<int, 3> clow, std::array<int, 3> chigh) :
        low(clow), high(chigh), size({high[0] - low[0] + 1, high[1] - low[1] + 1, high[2] - low[2] + 1}), order({0, 1, 2})
    {}
    //! \brief Constructs a box from the low and high indexes, also sets an order.
    box3d(std::array<int, 3> clow, std::array<int, 3> chigh, std::array<int, 3> corder) :
        low(clow), high(chigh), size({high[0] - low[0] + 1, high[1] - low[1] + 1, high[2] - low[2] + 1}), order(corder)
    {}
    //! \brief Returns true if the box contains no indexes.
    bool empty() const{ return (size[0] <= 0 or size[1] <= 0 or size[2] <= 0); }
    //! \brief Counts all indexes in the box, i.e., the volume.
    int count() const{ return (empty()) ? 0 : (size[0] * size[1] * size[2]); }
    //! \brief Creates a box that holds the intersection of this box and the \b other.
    box3d collide(box3d const other) const{
        return box3d({std::max(low[0], other.low[0]), std::max(low[1], other.low[1]), std::max(low[2], other.low[2])},
                     {std::min(high[0], other.high[0]), std::min(high[1], other.high[1]), std::min(high[2], other.high[2])}, order);
    }
    //! \brief Returns the box that is reduced in the given dimension according to the real-to-complex symmetry.
    box3d r2c(int dimension) const{
        switch(dimension){
            case 0: return box3d(low, {low[0] + size[0] / 2, high[1], high[2]}, order);
            case 1: return box3d(low, {high[0], low[1] + size[1] / 2, high[2]}, order);
            default: // dimension == 2
                return box3d(low, {high[0], high[1], low[2] + size[2] / 2}, order);
        }
    }
    //! \brief Compares two boxes, ignoring the order, returns \b true if all sizes and boundaries match.
    bool operator == (box3d const &other) const{
        return not (*this != other);
    }
    //! \brief Compares two boxes, ignoring the order, returns \b true if either of the box boundaries do not match.
    bool operator != (box3d const &other) const{
        for(int i=0; i<3; i++)
            if (low[i] != other.low[i] or high[i] != other.high[i]) return true;
        return false;
    }
    //! \brief Compares the order of two boxes, ignores the dimensions, returns \b true if the order is the same.
    bool ordered_same_as(box3d const &other) const{
        return (order[0] == other.order[0]) and (order[1] == other.order[1]) and (order[2] == other.order[2]);
    }
    //! \brief Get the ordered size of the dimension, i.e., size[order[dimension]].
    int osize(int dimension) const{ return size[order[dimension]]; }
    //! \brief The three lowest indexes.
    std::array<int, 3> const low;
    //! \brief The three highest indexes.
    std::array<int, 3> const high;
    //! \brief The number of indexes in each direction.
    std::array<int, 3> const size;
    //! \brief The order of the dimensions in the k * plane_stride + j * line_stride + i indexing.
    std::array<int, 3> const order;
};

/*!
 * \ingroup fft3dmisc
 * \brief Debugging info, writes out the box to a stream.
 */
inline std::ostream & operator << (std::ostream &os, box3d const box){
    for(int i=0; i<3; i++)
        os << box.low[i] << "  " << box.high[i] << "  (" << box.size[i] << ")\n";
    os << "(" << box.order[0] << "," << box.order[1] << "," << box.order[2] << ")\n";
    os << "\n";
    return os;
}

/*!
 * \ingroup fft3dbackend
 * \brief Return the number of 1-D ffts contained in the box in the given dimension.
 */
inline int fft1d_get_howmany(box3d const box, int const dimension){
    if (dimension == box.order[0]) return box.osize(1) * box.osize(2);
    if (dimension == box.order[1]) return box.osize(0);
    return box.osize(0) * box.osize(1);
}
/*!
 * \ingroup fft3dbackend
 * \brief Return the stride of the 1-D ffts contained in the box in the given dimension.
 */
inline int fft1d_get_stride(box3d const box, int const dimension){
    if (dimension == box.order[0]) return 1;
    if (dimension == box.order[1]) return box.osize(0);
    return box.osize(0) * box.osize(1);
}

/*!
 * \ingroup fft3dgeometry
 * \brief Pair of lists of input-output boxes as used by the heffte::fft3d.
 *
 * The strict contains all inboxes and outboxes for the ranks associated with the geometry of a heffte::fft3d transformation.
 */
struct ioboxes{
    //! \brief Inboxes for all ranks across the comm.
    std::vector<box3d> in;
    //! \brief Outboxes for all ranks across the comm.
    std::vector<box3d> out;
};

/*!
 * \ingroup fft3dgeometry
 * \brief Returns the box that encapsulates all other boxes.
 *
 * Searches through the world.in boxes and computes the highest and lowest of all entries.
 *
 * \param boxes the collection of all input and output boxes.
 */
inline box3d find_world(std::vector<box3d> const &boxes){
    std::array<int, 3> low  = boxes[0].low;
    std::array<int, 3> high = boxes[0].high;
    for(auto b : boxes){
        for(int i=0; i<3; i++)
            low[i] = std::min(low[i], b.low[i]);
        for(int i=0; i<3; i++)
            high[i] = std::max(high[i], b.high[i]);
    }
    return {low, high};
}

/*!
 * \ingroup fft3dgeometry
 * \brief Compares two vectors of boxes, returns true if all boxes match.
 */
inline bool match(std::vector<box3d> const &shape0, std::vector<box3d> const &shape1){
    if (shape0.size() != shape1.size()) return false;
    for(size_t i=0; i<shape0.size(); i++)
        if (shape0[i] != shape1[i])
            return false;
    return true;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Returns true if the geometry of the world is as expected.
 *
 * Runs simple checks to ensure that the inboxes will fill the world.
 * \param boxes is the collection of all world boxes
 * \param world the box that incorporates all other boxes
 *
 * The check is not very rigorous at the moment, a true rigorous test
 * will probably be too expensive unless lots of thought is put into it.
 */
inline bool world_complete(std::vector<box3d> const &boxes, box3d const world){
    long long wsize = 0;
    for(auto b : boxes) wsize += b.count();
    if (wsize < world.count())
        throw std::invalid_argument("The provided input boxes do not fill the world box!");

    for(size_t i=0; i<3; i++)
        if (world.low[i] != 0)
            throw std::invalid_argument("Global box indexing must start from 0!");

    for(size_t i=0; i<boxes.size(); i++)
        for(size_t j=0; j<boxes.size(); j++)
            if (i != j and not boxes[i].collide(boxes[j]).empty())
                throw std::invalid_argument("Input boxes cannot overlap!");

    return true;
}

/*!
 * \brief fft3dmisc
 * \brief Generate all possible factors of a number.
 *
 * Taking an integer \b n, generates a list of all possible way that
 * \b n can be split into a product of two numbers.
 * This is using a brute force method.
 *
 * \param n is the number to factor
 * \returns list of integer factors that multiply to \b n, i.e.,
 * \code
 *  std::vector<std::array<int, 2>> factors = get_factors(n);
 *  for(auto f : factors) assert( f[0] * f[1] == n );
 * \endcode
 *
 * Note: the result is never empty as it always contains (1, n).
 */
inline std::vector<std::array<int, 2>> get_factors(int const n){
    std::vector<std::array<int, 2>> result;
    for(int i=1; i<=n/2; i++){
        if (n % i == 0)
            result.push_back({i, n / i});
    }
    if (n == 1) {
        result.push_back({1, 1});
    }
    return result;
}

/*!
 * \ingroup fft3dmisc
 * \brief Factorize the MPI ranks into a 2D grid.
 *
 * Considers all possible factorizations of the total number of processors
 * and select the one with the lowest area which heuristically reduces the
 * number of ranks that need to communicate in each of the all-to-all operations.
 *
 * \param num_procs is the total number of processors to factorize
 *
 * \returns the two dimensions of the grid
 */
inline std::array<int, 2> make_procgrid(int const num_procs){
    auto factors = get_factors(num_procs);
    auto area = [](std::array<int, 2> f)->int{ return f[0]*f[1] + f[0] + f[1]; };
    std::array<int, 2> min_array = factors.front();
    int min_area = area(min_array);
    for(auto f : factors){
        int farea = area(f);
        if (farea < min_area){
            min_array = f;
            min_area = farea;
        }
    }
    return min_array;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Splits the world box into a set of boxes that will be assigned to a process in the process grid.
 *
 * \param world is a box describing all indexes of consideration,
 *              here there is no assumption on the lower or upper bound of the world
 * \param proc_grid describes a the number of boxes in each dimension
 *
 * \returns a list of non-overlapping boxes with union that fills the world where each box contains
 *          approximately the same number of indexes
 */
inline std::vector<box3d> split_world(box3d const world, std::array<int, 3> const proc_grid){

    auto fast = [=](int i)->int{ return world.low[0] + i * (world.size[0] / proc_grid[0]) + std::min(i, (world.size[0] % proc_grid[0])); };
    auto mid  = [=](int i)->int{ return world.low[1] + i * (world.size[1] / proc_grid[1]) + std::min(i, (world.size[1] % proc_grid[1])); };
    auto slow = [=](int i)->int{ return world.low[2] + i * (world.size[2] / proc_grid[2]) + std::min(i, (world.size[2] % proc_grid[2])); };

    std::vector<box3d> result;
    result.reserve(proc_grid[0] * proc_grid[1] * proc_grid[2]);
    for(int k = 0; k < proc_grid[2]; k++){
        for(int j = 0; j < proc_grid[1]; j++){
            for(int i = 0; i < proc_grid[0]; i++){
                result.push_back(box3d({fast(i), mid(j), slow(k)}, {fast(i+1)-1, mid(j+1)-1, slow(k+1)-1}, world.order));
            }
        }
    }
    return result;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Returns true if the shape forms pencils in the given direction.
 */
inline bool is_pencils(box3d const world, std::vector<box3d> const &shape, int direction){
    for(auto s : shape)
        if (s.size[direction] != world.size[direction])
            return false;
    return true;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Returns true if the shape forms slabs in the given directions.
 */
inline bool is_slab(box3d const world, std::vector<box3d> const &shape, int direction1, int direction2){
    for(auto s : shape)
        if (s.size[direction1] != world.size[direction1] or s.size[direction2] != world.size[direction2])
            return false;
    return true;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Returns the same shape, but sets a different order for each box.
 */
inline std::vector<box3d> reorder(std::vector<box3d> const &shape, std::array<int, 3> order){
    std::vector<box3d> result;
    result.reserve(shape.size());
    for(auto const &b : shape)
        result.push_back(box3d(b.low, b.high, order));
    return result;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Shuffle the new boxes to maximize the overlap with the old boxes
 *
 * A reshape operation from an old to a new configuration will require as much
 * MPI communication as the lack of overlap between the two box sets,
 * hence using a heuristic algorithms, in an attempt to find a reordering
 * of the boxes to increase the overlap with the old and new.
 * Also assigns the given order to the result.
 *
 * \param new_boxes is a set of new boxes to be reordered
 * \param old_boxes is the current box configuration
 * \param order is the new order to be assigned to the result boxes
 */
inline std::vector<box3d> maximize_overlap(std::vector<box3d> const &new_boxes,
                                           std::vector<box3d> const &old_boxes,
                                           std::array<int, 3> const order){
    std::vector<box3d> result;
    result.reserve(new_boxes.size());
    std::vector<bool> taken(new_boxes.size(), false);

    for(size_t i=0; i<new_boxes.size(); i++){
        // for each box in the result, find the box among the new_boxes
        // that has not been taken and has the largest overlap with the corresponding old box
        int max_overlap = -1;
        size_t max_index = new_boxes.size();
        for(size_t j=0; j<new_boxes.size(); j++){
            int overlap = old_boxes[i].collide(new_boxes[j]).count();
            if (not taken[j] and overlap > max_overlap){
                max_overlap = overlap;
                max_index = j;
            }
        }
        assert( max_index < new_boxes.size() ); // if we found a box
        taken[max_index] = true;
        result.push_back(box3d(new_boxes[max_index].low, new_boxes[max_index].high, order));
    }

    return result;
}

/*!
 * \ingroup fft3dgeometry
 * \brief Breaks the wold into a grid of pencils and orders the pencils to the ranks that will minimize communication
 *
 * A pencil is a box with one dimension that matches the entire world,
 * a pencils grid is a two dimensional grid of boxes that captures a three dimensional world box.
 *
 * This calls heffte::split_world() and then rearranges the list so that performing a reshape operation
 * from the source to the resulting list will minimize communication.
 * \param world is a box describing all indexes of consideration,
 *              it is assumed that the world is the union of the \b source boxes
 * \param prod_grid gives the number of boxes to use for the two-by-two grid
 * \param dimension is 0, 1, or 2, indicating the direction of orientation of the pencils,
 *                  e.g., dimension 1 means that pencil.size[1] == world.size[1]
 *                  for each pencil in the output list
 * \param source is the current distribution of boxes across MPI ranks,
 *               and will be used as a reference when remapping boxes to ranks
 * \param order is the box index order (fast, mid, slow) that will be assigned to the result.
 *
 * \returns a sorted list of non-overlapping pencils which union is the \b world box
 */
inline std::vector<box3d> make_pencils(box3d const world,
                                       std::array<int, 2> const proc_grid,
                                       int const dimension,
                                       std::vector<box3d> const &source,
                                       std::array<int, 3> const order
                                      ){
    // trivial case, the grid is already in a pencil format
    if (is_pencils(world, source, dimension))
        return reorder(source, order);

    // create a list of boxes ordered in column major format (following the proc_grid box)
    std::vector<box3d> pencils;
    if (dimension == 0){
        pencils = split_world(world, {1, proc_grid[0], proc_grid[1]});
    }else if (dimension == 1){
        pencils = split_world(world, {proc_grid[0], 1, proc_grid[1]});
    }else{ // third dimension
        pencils = split_world(world, {proc_grid[0], proc_grid[1], 1});
    }

    return maximize_overlap(pencils, source, order);
}

/*!
 * \ingroup fft3dgeometry
 * \brief Breaks the wold into a set of slabs that span the given dimensions
 *
 * The method is near identical to make_pencils, but the slabs span two dimensions.
 */
inline std::vector<box3d> make_slabs(box3d const world, int num_slabs,
                                     int const dimension1, int const dimension2,
                                     std::vector<box3d> const &source,
                                     std::array<int, 3> const order
                                     ){
    assert( dimension1 != dimension2 );
    std::vector<box3d> slabs;
    if (dimension1 == 0){
        if (dimension2 == 1){
            slabs = split_world(world, {1, 1, num_slabs});
        }else{
            slabs = split_world(world, {1, num_slabs, 1});
        }
    }else if (dimension1 == 1){
        if (dimension2 == 0){
            slabs = split_world(world, {1, 1, num_slabs});
        }else{
            slabs = split_world(world, {num_slabs, 1, 1});
        }
    }else{ // third dimension
        if (dimension2 == 0){
            slabs = split_world(world, {1, num_slabs, 1});
        }else{
            slabs = split_world(world, {num_slabs, 1, 1});
        }
    }

    return maximize_overlap(slabs, source, order);
}

/*!
 * \ingroup fft3dgeometry
 * \brief Creates a grid of mpi-ranks that will minimize the area of each of the boxes.
 *
 * Given the world box of indexes, generate the dimensions of a 3d grid of mpi-ranks,
 * where the grid is chosen to minimize the total surface area of each of the boxes.
 *
 * \param world is the box of all indexes starting from 0.
 * \param num_procs is the total number of mpi-ranks to use for the process grid.
 *
 * \returns the dimensions of the 3d grid that will minimize the size of each box.
 */
inline std::array<int, 3> proc_setup_min_surface(box3d const world, int num_procs){
    assert(world.count() > 0); // make sure the world is not empty

    // using valarrays that work much like vectors, but can perform basic
    // point-wise operations such as addition, multiply, and division
    std::valarray<int> all_indexes = {world.size[0], world.size[1], world.size[2]};
    // set initial guess, probably the worst grid but a valid one
    std::valarray<int> best_grid = {1, 1, num_procs};

    // internal helper method to compute the surface
    auto surface = [&](std::valarray<int> const &proc_grid)->
        int{
            auto box_size = all_indexes / proc_grid;
            return ( box_size * box_size.cshift(1) ).sum();
        };

    int best_surface = surface({1, 1, num_procs});

    for(int i=1; i<=num_procs; i++){
        if (num_procs % i == 0){
            int const remainder = num_procs / i;
            for(int j=1; j<=remainder; j++){
                if (remainder % j == 0){
                    std::valarray<int> candidate_grid = {i, j, remainder / j};
                    int const candidate_surface = surface(candidate_grid);
                    if (candidate_surface < best_surface){
                        best_surface = candidate_surface;
                        best_grid    = candidate_grid;
                    }
                }
            }
        }
    }

    assert(best_grid[0] * best_grid[1] * best_grid[2] == num_procs);

    return {best_grid[0], best_grid[1], best_grid[2]};
}

namespace mpi {
/*!
 * \ingroup hefftempi
 * \brief Gather all boxes across all ranks in the comm.
 *
 * Constructs an \b ioboxes struct with input and output boxes collected from all ranks.
 * \param my_inbox is the input box on this rank
 * \param my_outbox is the output box on this rank
 * \param comm is the communicator with all ranks
 *
 * \returns an \b ioboxes struct that holds all boxes across all ranks in the comm
 *
 * Uses MPI_Allgather().
 */
inline ioboxes gather_boxes(box3d const my_inbox, box3d const my_outbox, MPI_Comm const comm){
    std::array<box3d, 2> my_data = {my_inbox, my_outbox};
    std::vector<box3d> all_boxes(2 * mpi::comm_size(comm), box3d({0, 0, 0}, {0, 0, 0}));
    MPI_Allgather(&my_data, 2 * sizeof(box3d), MPI_BYTE, all_boxes.data(), 2 * sizeof(box3d), MPI_BYTE, comm);
    ioboxes result;
    for(auto i = all_boxes.begin(); i < all_boxes.end(); i += 2){
        result.in.push_back(*i);
        result.out.push_back(*(i+1));
    }
    return result;
}

}

}


#endif /* HEFFTE_GEOMETRY_H */
