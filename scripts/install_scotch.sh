# Check CC is defined
if [ -z "$CC" ]; then
  echo "Error: you need to define the CC environment variable (the C compiler)" 
  exit 1
fi
export MPICC=mpicc

# Check prefix is defined
if [ -z "$1" ]; then
  echo "Error: you must give a prefix on where to install scotch"
  echo 'Usage: ./install_scotch.sh $INSTALL_PREFIX [$N_PROC]'
  exit 1
fi
INSTALL_PREFIX=$1

# Get number of proc for make, or 1 if nothing given
if [ -z "$2" ]; then 
  N_PROC=1
else
  N_PROC=$2
fi

if [[ ! -d "scotch" ]]
then
  git clone --recursive https://gitlab.inria.fr/scotch/scotch.git
  (cd scotch && git checkout v6.0.9)
fi

mkdir -p $INSTALL_PREFIX

cd scotch/src
cat - > Make.inc/Makefile.inc.shlib <<EOF
EXE       =
LIB       = .so
OBJ       = .o
MAKE      = make
AR        = $CC
ARFLAGS   = -shared -o
CAT       = cat
CCS       = $CC
CCP       = $MPICC
CCD       = $MPICC
CFLAGS    = -O3 -Drestrict='' -DCOMMON_PTHREAD -DCOMMON_RANDOM_FIXED_SEED -DSCOTCH_RENAME -DIDXSIZE32
CLIBFLAGS = -fPIC -shared
LDFLAGS   = -lz -lm -lrt -pthread
CP        = cp
LEX       = flex -Pscotchyy -olex.yy.c
LN        = ln
MKDIR     = mkdir
MV        = mv
RANLIB    = echo
YACC      = bison -pscotchyy -y -b y
EOF

rm Makefile.inc; ln -sf Make.inc/Makefile.inc.shlib Makefile.inc
make -j$N_PROC esmumps
make -j$N_PROC ptesmumps
make -j$N_PROC scotch
make -j$N_PROC ptscotch
make prefix=$INSTALL_PREFIX install
