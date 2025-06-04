      subroutine vuanisohyper_inv (
C Read only -
     *     nblock, nFiber, nInv, 
     *     jElem, kIntPt, kLayer, kSecPt, 
     *     cmname,
     *     nstatev, nfieldv, nprops,
     *     props, tempOld, tempNew, fieldOld, fieldNew,
     *     stateOld, sInvariant, zeta,  
C     Write only -
     *     uDev, duDi, d2uDiDi,
     *     stateNew )
C
      include 'vaba_param_dp.inc'
C
      dimension props(nprops), 
     *  tempOld(nblock),
     *  fieldOld(nblock,nfieldv), 
     *  stateOld(nblock,nstatev), 
     *  tempNew(nblock),
     *  fieldNew(nblock,nfieldv),
     *  sInvariant(nblock,nInv), 
     *  zeta(nblock,nFiber*(nFiber-1)/2),
     *  uDev(nblock), duDi(nblock,*), 
     *  d2uDiDi(nblock,*),
     *  stateNew(nblock,nstatev)
C
      character*80 cmname
C
      parameter ( half = 0.5d0,
     *            zero = 0.d0, 
     *            one  = 1.d0, 
     *            two  = 2.d0, 
     *            three= 3.d0, 
     *            four = 4.d0, 
     *            five = 5.d0, 
     *            six  = 6.d0,
c
     *            index_I1 = 1,
     *            index_J  = 3,
     *            asmall   = 2.d-16  )
C
C     3-family fibre model (based on 4-family by Hu, Baek and Humphrey)
C
C     Read material properties
      C10 = props(1)
      rk1 = props(3)
      rk2 = props(4)
      rk3 = props(5)
	  rk4 = props(6)
c
      do k = 1, nblock
        uDev(k) = zero
C     Isotropic part of function		
		uDev(k) = uDev(k) + C10 * (sInvariant(k,index_I1) - three)
		duDi(k,index_I1) = C10
		d2uDiDi(k,indx(index_I1,index_I1)) = zero
C		
C	  Fibre contribution 
C
        do k1 = 1, nFiber
          index_i4 = indxInv4(k1,k1)
		  if(kl.gt.one) then
		    E_alpha1 = (sInvariant(k,index_i4) - one  )
			E_alpha = max(E_alpha1, zero)
		    aux     = exp(rk4*E_alpha*E_alpha)
C
C     deviatoric energy
			uDev(k) = uDev(k) +  (rk3 / (four* rk4) * ( aux - one ) )
C			
C	  duDi	
            duDi(k,index_i4) = ( (rk3 / 2) * aux ) * E_alpha
C
C     d2uDiDi
            d2uDiDi(k,indx(index_i4,index_i4)) = aux * ( rk3 * rk4 * E_alpha*E_alpha + one)
		  else
		    E_alpha1 = (sInvariant(k,index_i4) - one  )
			E_alpha = max(E_alpha1, zero)
		    aux     = exp(rk2*E_alpha*E_alpha)
c
c     deviatoric energy
c
			uDev(k) = uDev(k) +  (rk1 / (four* rk2) * ( aux - one ) )
C			
C	  duDi	
            duDi(k,index_i4) = ( (rk1 / 2) * aux ) * E_alpha
C
C     d2uDiDi
            d2uDiDi(k,indx(index_i4,index_i4)) = aux * ( rk1 * rk2 * E_alpha*E_alpha + one)
		  end if
        end do
      end do
c     
c     compressible case
      if(props(2).gt.zero) then
        do k = 1,nblock
          Dinv = one / props(2)
          det = sInvariant(k,index_J)
          duDi(k,index_J) = Dinv * (det - one/det)
          d2uDiDi(k,indx(index_J,index_J))= Dinv * (one + one / det / det)
        end do
      end if
c
      return
      end
C-------------------------------------------------------------
C     Function to map index from SquDevre to Triangular storage 
C 		 of symmetric matrix
C
      integer function indx( i, j )
      include 'vaba_param_dp.inc'
      ii = min(i,j)
      jj = max(i,j)
      indx = ii + jj*(jj-1)/2
      return
      end
C-------------------------------------------------------------
C
C     Function to generate enumeration of scalar
C     Pseudo-Invariants of type 4

      integer function indxInv4( i, j )
      include 'vaba_param_dp.inc'
      ii = min(i,j)
      jj = max(i,j)
      indxInv4 = 4 + jj*(jj-1) + 2*(ii-1)
      return
      end
C-------------------------------------------------------------
C
C     Function to generate enumeration of scalar
C     Pseudo-Invariants of type 5
C
      integer function indxInv5( i, j )
      include 'vaba_param_dp.inc'
      ii = min(i,j)
      jj = max(i,j)
      indxInv5 = 5 + jj*(jj-1) + 2*(ii-1)
      return
      end